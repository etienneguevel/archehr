from enum import Enum

import torch
from transformers import AutoTokenizer, AutoModel


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    DISTRIBUTED = "distributed"

def load_model_hf(model_name: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device == DeviceType.DISTRIBUTED:       
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
        )

    elif device == DeviceType.GPU:
        # Load the model and tokenizer
        try:
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                device_map="cuda",
            )

        except torch.OutOfMemoryError as e:
            print(f"Out of memory error: {e}")
            print("Using distribute_weights=True\n")
            
            return load_model_hf(model_name, device="distributed")
        
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    else:
        # Load the model and tokenizer
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="cpu",
        )

    return model, tokenizer

