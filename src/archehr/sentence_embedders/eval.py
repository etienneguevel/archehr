from typing import List, Tuple, Any, Dict

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from archehr.data.utils import load_data


def load_model(model_name: str)-> Tuple[nn.Module, AutoTokenizer]:
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

def make_queries(
    case: Dict[str, Any],
    model_name: str,
) -> List[str]:
    
    if model_name == 'intfloat/multilingual-e5-large':
        # Make the queries
        narrative_query = f"""
        Instruct: Given the following clinical narrative, indicate if the phrase
        is relevant to answer it.
        Query: {case['narrative']}
        """
        
        patient_question_query = f"""
        Instruct: Given the following patient question, retrieve if the phrase
        is relevant to answer it.
        Query: {case['patient_question']}
        """

        clinical_question_query = f"""
        Instruct: Given the following clinical question, retrieve if the phrase
        is relevant to answer it.
        Query: {case['clinical_question']}
        """

        queries = [
            narrative_query,
            patient_question_query,
            clinical_question_query,
        ]

        phrases = [f"Phrase: {s}" for _, s, _ in case['sentences']]

        # Retrieve the answers
        answers = [a for _, _, a in case['sentences']]

        return (queries + phrases), answers
    
    else:
        raise ValueError(f"Model {model_name} not supported.")

def average_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def main(
    model_name: str,
    data_path: str
):
    
    # Load the model and tokenizer
    model, tokenizer = load_model(model_name)

    # Load the data
    data = load_data(data_path)

    # Make the queries
    for case in data.values():
        queries, answers = make_queries(case, model_name)

        # Tokenize the queries
        inputs = tokenizer(
            queries,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Get the model outputs
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Average pool the last hidden states
        embeddings = average_pool(
            outputs.last_hidden_state,
            inputs['attention_mask']
        )
