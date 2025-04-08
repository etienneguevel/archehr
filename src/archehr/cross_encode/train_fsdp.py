import json
import os
from argparse import ArgumentParser, Namespace

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
)

from archehr.eval import do_eval
from archehr.cross_encode.nli_deberta import remove_last_layer
from archehr.data.utils import load_data, make_query_sentence_pairs
from archehr.data.dataset import QADataset


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
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
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
    model = remove_last_layer(model)
    model.to(device)

    # Wrap the model with FSDP
    model = wrap(model)
    model = FSDP(model)

    # Make the queries
    pairs_train = make_query_sentence_pairs(data_train)
    pairs_val = make_query_sentence_pairs(data_val)

    # Create the dataset
    dataset_train = QADataset(pairs_train, tokenizer)
    dataset_val = QADataset(pairs_val, tokenizer)

    # Create the dataloader
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    dataloader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize the metrics list
    metrics_list = []
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            os.path.join(save_folder, f'profile_{torch.distributed.get_rank()}'),
        ),
        with_stack=True,
    ) as prof:
        for epoch in (progress_bar := tqdm(range(num_epochs))):
            model.train()
            for batch in dataloader_train:
                # Move inputs and labels to device
                batch = {k: v.to(device) for k, v in batch.items()}

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(**batch)

                # Backward pass and optimization
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            prof.step()  # Step the profiler

            # Validation loop
            if epoch % 10 == 0:
                metrics = do_eval(
                    model,
                    dataloader_val,
                    device,
                    loss,
                    target=dataset_val.translate_dict['essential'],
                    progress_bar=progress_bar,
                )
                metrics_list.append(metrics)

    # Save the metrics
    if torch.distributed.get_rank() == 0:  # Save only on rank 0
        metrics_path = os.path.join(save_folder, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_list, f, indent=4)

    # Save the model
    if torch.distributed.get_rank() == 0:  # Save only on rank 0
        model.save_pretrained(os.path.join(save_folder, 'model'))

    # Cleanup distributed process group
    destroy_process_group()

    return model


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
