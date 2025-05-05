import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
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
    note_excerpt = patient_case.find('note_excerpt').text

    return [
    (
    f'Instruct: You are given a question from a patient:\n {patient_narrative}\n'
    f'Which has been reformulated by a clinician: {clinician_question}\n'
    # f'As well as a detailed report about his medical trajectory in a xml format:' 
    # f'{note_excerpt}\n'
    f'Query: is sentence: {sentence.text}\nrelevant for the question ?'
    )
        for i, sentence in enumerate(
            patient_case.find('note_excerpt_sentences').findall('sentence')
        )
    ]

def translate_xlm_pandas(file_path):
    # Load the data
    root, labels = load_data(file_path)

    # Make it into dataframe
    data = []
    for i, (c, label) in enumerate(zip(root.findall('case'), labels, strict=True)):
        case_data = [
            [
                i,
                c.find("note_excerpt").text,
                c.find("clinician_question").text,
                j,
                sent.text,
                lab["relevance"]
            ]
            for j, (sent, lab) in enumerate(
                zip(c.find("note_excerpt_sentences").findall("sentence"), label["answers"])
            )
        ]
        data.extend(case_data)

    return pd.DataFrame(data, columns=["case_id", "note_excerpt", "question_generated", "sentence_id", "ref_excerpt", "relevance"])

def make_instruction(row):
    instruction = (
        "Instruct: Given a patient medical history, a medical question and a phrase"
        "from the medical history, determine if this phrase is relevant to answer the"
        " question.\n"
    )

    query = (
        f"Query: \nMedical history of the patient:\n{row.note_excerpt}\n"
        f"Question:\n{row.question_generated}\nPhrase:\n{row.ref_excerpt}"
    )

    return instruction + query

def make_augmented_dataset(data_path, augmented_data_path):
    # Load data
    df_data = translate_xlm_pandas(data_path)
    df_data["source"] = "archehr"
    df_augmented = pd.read_excel(augmented_data_path)
    df_augmented["source"] = "augmented"
    
    # Split train / val
    df_train = pd.concat([
        df_data[df_data.case_id < 15],
        df_augmented
    ])

    df_val = df_data[~(df_data.case_id < 15)]

    # Make the prompts
    df_train["text"] = df_train.apply(make_instruction, axis=1)
    df_val["text"] = df_val.apply(make_instruction, axis=1)

    # Translate the labels
    df_train["labels"] = df_train["relevance"].apply(
        lambda x: TRANSLATE_DICT.get(x)
    )
    df_val["labels"] = df_val["relevance"].apply(
        lambda x: TRANSLATE_DICT.get(x)
    )

    # Return in hf format
    df_train = Dataset.from_pandas(df_train)
    df_val = Dataset.from_pandas(df_val)

    return df_train, df_val

def make_hf_dict(
    root,
    labels,
    split=0.2,
):
    # Create the train / eval dicts
    train_dict = {
        'case': [],
        'text': [],
        'labels': []
    }

    eval_dict = {
        'case': [],
        'text': [],
        'labels': []
    }

    # Find the split
    n_cases = len(root.findall('case'))
    sep = int((1 - split) * n_cases)


    for i, (c, labs) in enumerate(zip(root.findall('case'), labels)):

        if i < sep:
            train_dict['case'].extend([i for _ in labs['answers']])
            train_dict['text'].extend(get_detailed_instruction(c))
            train_dict['labels'].extend(
                [TRANSLATE_DICT.get(a['relevance'], 1)] for a in labs['answers']
            )
        
        else:
            eval_dict['case'].extend([i for _ in labs['answers']])
            eval_dict['text'].extend(get_detailed_instruction(c))
            eval_dict['labels'].extend(
                [TRANSLATE_DICT.get(a['relevance'], 1)] for a in labs['answers']
            )
        
    train_dataset = Dataset.from_dict(train_dict)
    eval_dataset = Dataset.from_dict(eval_dict)
    
    return train_dataset, eval_dataset

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