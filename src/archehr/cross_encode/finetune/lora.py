import warnings
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial

import numpy as np
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
)
from transformers.trainer_utils import EvalPrediction

from archehr import PROJECT_DIR
from archehr.utils.loaders import DeviceType
from archehr.data.utils import load_data, make_augmented_dataset, TRANSLATE_DICT
from archehr.models.qwen2.Qwen2 import Qwen2EmbClassification


BASE_CONFIG = PROJECT_DIR / "src/archehr/configs/finetune/base_config.yaml"

def get_args_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--folder_name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--config_path",
        default=BASE_CONFIG,
        type=str,
    )

    return parser.parse_args()

def compute_metrics(
    eval_pred: EvalPrediction,
    target_class: str | int
):
    # Unpack eval_pred    
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Calculate correct pred, total amount
    
    logits = np.array(logits)
    preds = np.argmax(logits, -1)
    labels = np.array(labels)
    if labels.ndim > 1:
        labels = labels[:, 0]

    total = labels.shape[0]
    correct = np.sum((preds == labels))

    # Calculate tp, fp, fn
    tp = np.sum((labels == target_class) & (preds == target_class))
    fp = np.sum((labels != target_class) & (preds == target_class))
    fn = np.sum((labels == target_class) & (preds != target_class))

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0
    )
    acc = correct / total

    return {
        "acc": round(acc, 3),
        "ppv": round(precision, 3),
        "rec": round(recall, 3),
        "f1": round(f1, 3),
    }


def _make_datasets(
    data_path: str,
):
    # Load the data
    file_path = data_path / "1.1" / "dev"
    augmented_data_path = data_path / "qa_results_structured.xlsx"
    train_dataset, eval_dataset = make_augmented_dataset(
        file_path, augmented_data_path
    )

    return train_dataset, eval_dataset

def _build_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    device: DeviceType,
    peft_config: LoraConfig,
):
    # Make the model
    # 1. Load base model
    model = Qwen2EmbClassification.from_pretrained(
        model_name,
        2,
        peft_config,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    return model, tokenizer

def _setup_trainer(
    model,
    train_dataset,
    val_dataset,
    folder_name: str,
    train_args,
):
    # Make the training config
    strategy = train_args.pop("strategy")

    training_args = TrainingArguments(
        output_dir=PROJECT_DIR / "models" / folder_name,
        eval_strategy=strategy,
        save_strategy=strategy,
        label_names=["labels"],
        load_best_model_at_end=True,
        logging_dir=PROJECT_DIR / "logs"/ folder_name,
        save_total_limit=3,
        **train_args
    )

    print("Training with:\n", training_args)

    # Make the metric fct
    metric_fct = partial(
        compute_metrics,
        target_class=TRANSLATE_DICT.get("essential")
    )

    # Make a custom callback to also evaluate the training metrics
    class CustomCallback(TrainerCallback):
        def __init__(self, trainer) -> None:
            super().__init__()
            self._trainer = trainer
        
        def on_epoch_end(self, args, state, control, **kwargs):
            if control.should_evaluate:
                control_copy = deepcopy(control)
                self._trainer.evaluate(
                    eval_dataset=self._trainer.train_dataset,
                    metric_key_prefix="train"
                )
                return control_copy

    # Build the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metric_fct
    )
    trainer.add_callback(CustomCallback(trainer))

    return trainer


def do_train(
    model_path: str,
    data_path: str,
    folder_name:str,
    lora_args,
    training_args,
    device: DeviceType = "auto",
):
    """
    Function made to be launched with accelerate : 
    `accelerate launch src/archehr/cross_encode/finetune/lora.py`
    """
    # Launch accelerate
    # accelerator = Accelerator()

    # Make the PEFT config
    target_modules = list(lora_args.pop("target_modules"))
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, #Sequence cls or Token cls ?
        inference_mode=False,
        target_modules=target_modules,
        **lora_args
    )
    
    # Build the model / tokenizer
    model, tokenizer = _build_model_and_tokenizer(
        model_path,
        2,
        device,
        peft_config
    )
    
    # Build the dataset
    dataset_train, dataset_val = _make_datasets(data_path)

    # Tokenizer the datasets
    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="longest",
            max_length=1024,
            truncation=True
        )

    train_dataset = dataset_train.map(tokenize, batched=True)
    train_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels",]
    )
    eval_dataset = dataset_val.map(tokenize, batched=True)
    eval_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels",]
    )
    print(f"Training dataset: {len(train_dataset)} elements.\n")
    print(f"Evaluation dataset: {len(eval_dataset)} elements.\n")

    # Make the trainer
    trainer = _setup_trainer(
        model,
        train_dataset,
        eval_dataset,
        folder_name,
        training_args,
    )

    # Launch training
    trainer.train()

    # Save the best model
    model = trainer.model
    model.save_pretrained(PROJECT_DIR / "final_model_qwen2")

def main():
    warnings.filterwarnings("ignore")
    args = get_args_parser()
    base_config = OmegaConf.load(BASE_CONFIG)
    config = OmegaConf.load(args.config_path)

    training_conf = OmegaConf.merge(base_config, config)

    do_train(
        model_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        folder_name=args.folder_name,
        data_path=PROJECT_DIR / "data",
        device="cpu",
        lora_args=training_conf.get("lora_args"),
        training_args=training_conf.get("training_args"),
    )


if __name__ == "__main__":
    main()
