import torch

from archehr.data.utils import to_device, get_labels

def do_eval(model, dataloader, device, loss, target, progress_bar=None):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate.
        dataloader: The dataloader for the validation set.
        device: The device to use for evaluation.
    
    Returns:
        The average loss and accuracy on the validation set.
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for batch in dataloader:
            # Move inputs and labels to device
            batch, labels = get_labels(batch)
            batch = to_device(batch, device)
            labels = labels.to(device)

            # Forward pass
            if isinstance(batch, dict):
                outputs = model(**batch).logits
            else:
                outputs = model(batch)

            # Compute the loss
            l_ = loss(outputs, labels)
            val_loss += l_.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Compute true positives & false positives & false negatives
            tp += sum((labels == target) & (predicted == target)).item()
            fp += sum((labels != target) & (predicted == target)).item()
            fn += sum((labels == target) & (predicted != target)).item()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0
    )

    avg_loss = val_loss / len(dataloader)
    accuracy = correct / total

    if progress_bar is not None:
        progress_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{accuracy:.1%}",
            ppv=f"{precision:.1%}",
            rec=f"{recall:.1%}",
            f1=f"{f1:.1%}",
        )

    output_dict = {
        'loss': avg_loss,
        'acc': accuracy,
        'ppv': precision,
        'rec': recall,
        'f1': f1,
    }

    return output_dict
