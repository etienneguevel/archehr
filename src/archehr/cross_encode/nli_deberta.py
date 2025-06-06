import torch.nn as nn


def remove_last_layer(model: nn.Module) -> nn.Module:
    """
    Remove the last layer of the model.
    
    Args:
        model (nn.Module): The model to modify.
    
    Returns:
        nn.Module: The modified model with the last layer removed.
    """
    # Assuming the last layer is a Linear layer
    if hasattr(model, 'classifier'):
        # Replace the last layer
        model.classifier = nn.Sequential(
            nn.Linear(model.config.hidden_size, 3),
            nn.Dropout(p=0.1)
        )

        # Set the parameters to require gradients
        for name, param in model.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False  

    else:
        raise ValueError("Model does not have a recognizable last layer.")

    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f'model has {num_trainable_params} trainable parameters.')
    
    return model
