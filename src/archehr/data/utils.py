import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import torch
from datasets import Dataset
from torch import Tensor


TRANSLATE_DICT = {
    'essential': 0,
    'not-relevant': 1,
    'supplementary': 1,
}

def load_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Load the data of the Archehr dataset.
    Args:
        data_path (str): Path to the dataset.
    Returns:
        Dict[str, Any]: Dictionary containing the data. Keys are the case ids,
        values are dictionaries containing the narrative, patient question, 
        clinical question, and sentences.
        Senteces are tuples containing the sentence id, sentence text, and 
        relevance.
    """
    case_file = 'archehr-qa.xml'
    json_file = 'archehr-qa_key.json'

    if not os.path.isdir(data_path):
        raise ValueError(f"Invalid directory: {data_path}")
    
    if case_file not in os.listdir(data_path):
        raise ValueError(f"Missing {case_file} in {data_path}")
    
    if json_file not in os.listdir(data_path):
        raise ValueError(f"Missing {json_file} in {data_path}")


    # Load the xml file
    tree = ET.parse(Path(data_path) / case_file)
    root = tree.getroot()

    # Load the json file
    with open(Path(data_path) / json_file, 'r') as f:
        labels = json.load(f)

    return root, labels

def make_query_sentence_pairs(
    root,
    labels,
    prompt_template: Optional[List[str]] = None,
):
    # Transform the xml data into a dictionary
    data = []
    for c, label in zip(root.findall('case'), labels, strict=True):
        data.append({
            'narrative': c.find('patient_narrative').text,
            'patient_question': c.find('patient_question').find('phrase').text,
            'clinician_question': c.find('clinician_question').text,
            'sentences': [
                (j, sentence.text, answer['relevance'])
                for j, (sentence, answer) in enumerate(
                    zip(
                        c.find('note_excerpt_sentences').findall('sentence'),
                        label['answers'],
                        strict=True
                    )
                )
            ],
        })

    prompt_options = [
        'narrative',
        'patient_question',
        'clinician_question',
    ]

    if not prompt_template:
        prompts = prompt_options

    else:
        for p in prompt_template:
            if p not in prompt_options:
                raise ValueError(f"Invalid prompt template: {p}")
        prompts = prompt_template
    
    # Create the query sentence pairs
    query_sentence_pairs = []
    for i, c in enumerate(data):
        
        for p in prompts:
            if p == 'narrative':
                query = c['narrative']

            elif p == 'patient_question':
                query = c['patient_question']

            elif p == 'clinician_question':
                query = c['clinician_question']

            for _, sentence, answer in c['sentences']:
                query_sentence_pairs.append(
                    {
                        'case_id': i,
                        'query': (query, sentence),

                        'label': answer
                    }
                )
    
    return query_sentence_pairs

def get_detailed_instruction(patient_case):

    patient_narrative = patient_case.find('patient_narrative').text
    clinician_question = patient_case.find('clinician_question').text
    note_excerpt = patient_case.find('note_excerpt_sentences').text

    return [
    f'''Instruct: You are given a question from a patient : {patient_narrative} which has been reformulated by a clinician {clinician_question}, as well as a detailed report about his medical trajectory in a xml format {note_excerpt}.
    Query: is sentence {i} relevant for the question ?
    '''
        for i, _ in enumerate(
            patient_case.find('note_excerpt_sentences').findall('sentence'),
        )
    ]

def make_hf_dict(
    root,
    labels
):
    
    output_dict = {
        'prompt': [],
        'labels': []
    }

    for c, labs in zip(root.findall('case'), labels):
        output_dict['text'].extend(get_detailed_instruction(c))
        output_dict['labels'].extend(
            TRANSLATE_DICT.get(a['relevance'], 1) for a in labs['answers']
        )
    
    return Dataset.from_dict(output_dict)

def last_token_pool(
    last_hidden_states: Tensor,
    attention_mask: Tensor
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(
            batch_size,
            device=last_hidden_states.device
        ), sequence_lengths]
    
def to_device(
    batch: Dict[str, Any] | Tensor,
    device: torch.device
) -> Dict[str, Tensor]:
    """
    Move the batch to the specified device.
    
    Args:
        batch (Dict[str, Any]): The batch to move.
        device (torch.device): The device to move the batch to.
    
    Returns:
        Dict[str, Tensor]: The batch on the specified device.
    """
    if isinstance(batch, Tensor):
        batch = batch.to(device)

    else:
        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

    return batch

def get_labels(batch):

    if isinstance(batch, dict):
        labels = batch.pop('labels')

    else:
        batch, labels = batch

    return batch, labels