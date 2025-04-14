import contextlib
import json
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from archehr.eval.eval import do_eval
from archehr.eval.mlp import Mlp
from archehr.data.utils import (
    load_data, make_query_sentence_pairs, to_device, get_labels
)
from archehr.data.dataset import QADatasetEmbedding
from archehr.utils.cluster import initialize_profiler


def parse_args() -> Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser(description="Train a model on the Archehr dataset.")
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="cross-encoder/nli-deberta-v3-base",
        help="Name of the pre-trained model to load."
    )
    
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the Archehr dataset."
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training."
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of epochs to train."
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer."
    )

    parser.add_argument(
        "--save_folder",
        type=str,
        required=True,
        help="Path of the folder where to save information."
    )
    
    parser.add_argument(
        "--world_size",
        type=int,
        default=1,
        help="Number of processes for distributed training."
    )
    
    return parser.parse_args()


def load_model(model_name: str) -> tuple[nn.Module, AutoTokenizer]:
    """
    Load a pre-trained model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
    
    Returns:
        tuple: A tuple containing the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    return model, tokenizer


def do_train(
    model_name: str,
    data_path: str,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    save_folder: str,
) -> None:
    """
    Train a model on the Archehr dataset using FSDP.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        save_folder (str): Path to save the trained model and metrics.
    """
    # Initialize distributed process group
    init_process_group(backend="nccl")
    
    # Get the rank of the current process
    rank = torch.distributed.get_rank()
    
    # Set the device for the current rank
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    print(f"Rank {rank} is using GPU {torch.cuda.current_device()}")

    # Load the data
    data = load_data(data_path)
    n_cases = len(data)

    # Split the data into training and validation sets
    data_train = data[:int(n_cases * 0.8)]
    data_val = data[int(n_cases * 0.8):]
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)
    model.to(device)
    model.eval()

    # Make the queries
    pairs_train = make_query_sentence_pairs(data_train)
    pairs_val = make_query_sentence_pairs(data_val)

    # Create the dataset

    dataset_train = QADatasetEmbedding(pairs_train, tokenizer, model)
    dataset_val = QADatasetEmbedding(pairs_val, tokenizer, model)

    # Create the dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
    )

    # Create the MLP
    mlp = Mlp(
        dataset_train.emb_size.numel(),
        out_features=len(dataset_train.translate_dict)
    ).to(device)

    # Wrap the model with FSDP
    mlp = wrap(mlp)
    mlp = FSDP(mlp, use_orig_params=True)
    
    num_trainable_params = sum([
        p.numel() for p in mlp.parameters() if p.requires_grad
    ])
    print(f'There are {num_trainable_params} trainable params\n\n')

    # Define the optimizer
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=learning_rate)
    loss = nn.CrossEntropyLoss()

    # Initialize the metrics list
    metrics_list = []

    # Initialise the profiler
    if rank == 0: 
        profiler = initialize_profiler(save_folder)
        torch.cuda.memory._record_memory_history()

    else:
        profiler = contextlib.nullcontext()

    with profiler as prof:
        for epoch in (progress_bar := tqdm(range(num_epochs))):
            mlp.train()
            for batch in dataloader_train:
                # Move inputs and labels to device
                batch, labels = get_labels(batch)

                batch = to_device(batch, device)
                labels = to_device(labels, device)

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                if isinstance(batch, dict):
                    outputs = mlp(**batch)
                else:
                    outputs = mlp(batch)

                # Backward pass and optimization
                l_ = loss(outputs, labels)
                l_.backward()
                optimizer.step()
            
            if rank == 0:         
                prof.step()  # Step the profiler

            # Validation loop
            if epoch % 10 == 0:
                metrics = do_eval(
                    mlp,
                    dataloader_val,
                    device,
                    loss,
                    target=dataset_val.translate_dict['essential'],
                    progress_bar=progress_bar,
                )
                metrics_list.append(metrics)

            if (epoch == 5) & (rank == 0):
                torch.cuda.memory._dump_snapshot(
                    os.path.join(save_folder, "memory_snapshot.pickle"),
                )

    # Save the metrics
    if torch.distributed.get_rank() == 0:  # Save only on rank 0
        metrics_path = os.path.join(save_folder, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_list, f, indent=4)

    # Save the model
    if torch.distributed.get_rank() == 0:  # Save only on rank 0
        mlp.save_pretrained(os.path.join(save_folder, 'model'))

    # Cleanup distributed process group
    destroy_process_group()

    return mlp


def main():
    """
    Main function to run the training script.
    """
    args = parse_args()
    # Set the save directory
    os.makedirs(args.save_folder, exist_ok=True)
    
    # Train the model
    _ = do_train(
        model_name=args.model_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_folder=args.save_folder,
    )

    
if __name__ == "__main__":
    main()
