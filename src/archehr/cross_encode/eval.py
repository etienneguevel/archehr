import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from archehr.data.utils import load_data, make_query_sentence_pairs, to_device, get_labels
from archehr.data.dataset import QADatasetEmbedding
from archehr.eval.eval import do_eval
from archehr.eval.basic_models import Mlp, Fc
from archehr.utils.cluster import get_idle_gpus

def parse_args():

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

    return parser

def make_datasets(data_path, model_name, device):
    data = load_data(data_path)
    n_cases = len(data)

    # Split the data into train and test sets
    data_train = data[:int(n_cases * 0.8)]
    data_val = data[int(n_cases * 0.8):]

    # Make the queries
    pairs_train = make_query_sentence_pairs(data_train)
    pairs_val = make_query_sentence_pairs(data_val)

    # Create the dataset
    dataset_train = QADatasetEmbedding(pairs_train, model_name, device)
    dataset_val = QADatasetEmbedding(pairs_val, model_name, device)
    
    return dataset_train, dataset_val

def do_train(
    model: nn.Module,
    dataset_train: QADatasetEmbedding,
    dataset_val: QADatasetEmbedding,
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    device: torch.device = torch.device('cpu')
):
    """
    Train a model on the Archehr dataset.

    Args:
        model (nn.Module): The model to train.
        dataset_train (QADatasetEmbedding): The training dataset.
        dataset_val (QADatasetEmbedding): The validation dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Set the device
    model.to(device)

    # Create the data loaders
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Define the optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    metrics_list = []
    for epoch in (progress_bar := tqdm(range(num_epochs))):
        model.train()
        for batch in train_loader:
            # Move inputs and labels to device
            batch, labels = get_labels(batch)
            batch = to_device(batch, device)
            labels = to_device(labels, device)

            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(batch, Tensor):
                outputs = model(batch)

            else:
                outputs = model(**batch)

            # Backward pass and optimization
            l_ = loss(outputs, labels)
            l_.backward()
            optimizer.step()
        
        # Validation loop
        if epoch % 10 == 0:
            metrics = do_eval(
                model,
                val_loader,
                device,
                loss,
                target=dataset_val.translate_dict['essential'],
                progress_bar=progress_bar,
            )
            metrics_list.append(metrics)

    # Print final metrics
    metrics_train = do_eval(
        model,
        train_loader,
        device,
        loss,
        target=dataset_train.translate_dict['essential'],
    )
    metrics_val = do_eval(
        model,
        val_loader,
        device,
        loss,
        target=dataset_val.translate_dict['essential'],
    )
    print(f"""
    Train | loss: {metrics_train['loss']:.4f} | acc: {metrics_train['acc']:.1%} 
    | ppv: {metrics_train['ppv']:.1%} | rec: {metrics_train['rec']:.1%} | 
    f1: {metrics_train['f1']:.1%}
    """)
    print(f"""
    Validation | loss: {metrics_val['loss']:.4f} | acc: {metrics_val['acc']:.1%}
    | ppv: {metrics_val['ppv']:.1%} | rec: {metrics_val['rec']:.1%} | 
    f1: {metrics_val['f1']:.1%}
    """)

    return model
    
def main():

    parser = parse_args()
    args = parser.parse_args()

    # Set the device
    idle_gpus = get_idle_gpus()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, idle_gpus))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the data
    dataset_train, dataset_val = make_datasets(
        args.data_path,
        args.model_name,
        device=device
    )

    # Create the models
    emb_dim = dataset_train.emb_size.numel()
    print("Embeddings dimension: ", emb_dim, "\n")
    mlp = Mlp(
        in_features=emb_dim,
        out_features=len(dataset_train.translate_dict),
    )

    fc = Fc(
        in_features=emb_dim,
        out_features=len(dataset_train.translate_dict),
    )

    for m in [mlp, fc]:
        m.to(device)
        # Train the model
        print(f"Training model: {m}")

        _ = do_train(
            m,
            dataset_train,
            dataset_val,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            device=device
        )

if __name__ == "__main__":
    main()
