from enum import Enum

import torch
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import AutoTokenizer, AutoModel


class DeviceType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    DISTRIBUTED = "distributed"

def load_model_hf(model_name: str, device: str):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if device == DeviceType.DISTRIBUTED:
        with init_empty_weights():
            # Load the model and tokenizer
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
        # Infer the device map
        n_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {n_gpus}\n")
        
        device_map = infer_auto_device_map(
            model,
            max_memory={i: "16GiB" for i in range(n_gpus)},
        )

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map=device_map,
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
