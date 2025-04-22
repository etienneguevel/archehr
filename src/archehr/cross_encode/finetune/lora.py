from functools import partial

import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import EvalPrediction

from archehr import PROJECT_DIR
from archehr.utils.loaders import DeviceType
from archehr.data.utils import load_data, make_hf_dict, TRANSLATE_DICT
from archehr.models.qwen2.Qwen2 import Qwen2EmbClassification


def compute_metrics(
    eval_pred: EvalPrediction,
    target_class: str | int
):
    # Unpack eval_pred    
    logits = eval_pred.predictions
    labels = eval_pred.label_ids

    # Safeguard for DTensors
    if hasattr(logits, "to_local"):
        logits = logits.to_local()

    if hasattr(labels, "to_local"):
        labels = labels.to_local()

    logits = np.array(logits)
    labels = np.array(labels)

    # Calculate correct pred, total amount
    preds = np.argmax(logits, axis=-1)
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
        "acc": acc,
        "ppv": precision,
        "rec": recall,
        "f1": f1,
    }


def _make_datasets(
    data_path: str,
):
    # Load the data
    root, labels = load_data(data_path)

    # Split train / val
    dataset = make_hf_dict(root, labels)
    split_dataset = dataset.train_test_split(test_size=0.2)
    dataset_train = split_dataset["train"]
    dataset_val = split_dataset["test"]

    return dataset_train, dataset_val

def _build_model_and_tokenizer(
    model_name: str,
    num_labels: int,
    device: DeviceType,
    peft_config: LoraConfig,
):
    # Load the model
    model = Qwen2EmbClassification.from_pretrained(
        model_name,
        num_labels,
        device_map=device,
        trust_remote_code=True,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    print(model)

    return model, tokenizer

def _setup_trainer(
    model,
    train_dataset,
    val_dataset,
):

    # Make the training config
    # TODO: make it in yaml file
    training_args = TrainingArguments(
        output_dir=PROJECT_DIR / "models/Qwen2",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=256,
        fp16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        optim="paged_adamw_32bit",
        label_names=["labels"],
        logging_steps=10,
        learning_rate=2e-4,
        load_best_model_at_end=True,
        logging_dir=PROJECT_DIR / "logs/Qwen2",
    )

    # Make the trainer
    metric_fct = partial(
        compute_metrics,
        target_class=TRANSLATE_DICT.get("essential")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metric_fct
    )

    return trainer


def do_train(
    model_path: str,
    data_path: str,
    device: DeviceType = "auto",
):
    """
    Function made to be launched with accelerate : 
    `accelerate launch src/archehr/cross_encode/finetune/lora.py`
    """
    # Launch accelerate
    # accelerator = Accelerator()

    # Make the PEFT config
    # TODO: make it in yaml file
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, #Sequence cls or Token cls ?
        inference_mode=False,
        r=8,
        target_modules=["q_proj", "v_proj"],
        lora_alpha=32,
        lora_dropout=0.1
    )
    

    # Build the model / tokenizer
    model, tokenizer = _build_model_and_tokenizer(
        model_path,
        2,
        device,
        peft_config
    )
    
    # Build the dataset
    # TODO: error in the shape of the dataset output tensors -> not enough dim
    dataset_train, dataset_val = _make_datasets(data_path)

    # Tokenizer the datasets
    def tokenize(example):
        return tokenizer(
            example["text"],
            max_length=8192,
            padding="max_length",
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

    # Make the trainer
    trainer = _setup_trainer(
        model,
        train_dataset,
        eval_dataset,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    do_train(
        model_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        data_path=PROJECT_DIR / "data" / "1.1" / "dev",
        device="cuda"
    )
