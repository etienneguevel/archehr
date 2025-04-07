import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from xml.etree import ElementTree as ET


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

    return data

def make_query_sentence_pairs(
    data,
    prompt_template: Optional[List[str]] = None,
):
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
