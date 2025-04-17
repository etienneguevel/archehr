import json
import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
)

from archehr import SAVE_DIR
from archehr.eval.eval import do_eval
from archehr.data.utils import load_data, make_query_sentence_pairs
from archehr.data.dataset import QADataset
from archehr.cross_encode.nli_deberta import remove_last_layer


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        Namespace: Parsed arguments.
    """
    parser = ArgumentParser (description="Train a model on the Archehr dataset.")
    
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
        "--save_name",
        type=str,
        default="model",
        help="Name to save the trained model."
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
    batch_size: int = 32,
    num_epochs: int = 10,
    learning_rate: float = 1e-5,
) -> None:
    """
    Train a model on the Archehr dataset.
    
    Args:
        model_name (str): The name of the pre-trained model to load.
        data_path (str): Path to the dataset.
        batch_size (int): Batch size for training.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
    """
    # Load the data
    data = load_data(data_path)
    n_cases = len(data)

    # Split the data into training and validation sets
    data_train = data[:int(n_cases * 0.8)]
    data_val = data[int(n_cases * 0.8):]
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)
    model = remove_last_layer(model)

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
    
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Initialize the metrics list
    metrics_list = []
    
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
    metrics_path = SAVE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_list, f, indent=4)

    return model

def main():
    """
    Main function to run the training script.
    """
    args = parse_args()
    
    # Train the model
    model = do_train(
        model_name=args.model_name,
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate
    )
    
    # Save the model
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR / args.save_name)
    
if __name__ == "__main__":
    main()
