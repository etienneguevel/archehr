from functools import partial

import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
from transformers.trainer_utils import EvalPrediction
from trl import SFTTrainer, SFTConfig

from archehr import PROJECT_DIR
from archehr.utils.loaders import load_model_hf, DeviceType
from archehr.data.utils import load_data, make_hf_dict


def compute_metrics(
    eval_pred: EvalPrediction,
    target_class: str | int
):
    # Unpack eval_pred    
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)

    # Calculate correct pred, total amount
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
    dataset_train = split_dataset['train']
    dataset_val = split_dataset['test']

    return dataset_train, dataset_val

def _build_model(
    model_name: str,
    device: DeviceType,
):
    # Load the model
    model, tokenizer = load_model_hf(
        model_name=model_name,
        device=device
    )

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
    
    # Adapt the model for peft
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())
    print(model)

    return model, tokenizer, peft_config

def _setup_trainer(
    model,
    tokenizer,
    peft_config,
    train_dataset,
    val_dataset,
):

    # Make the training config
    # TODO: make it in yaml file
    training_args = SFTConfig(
        output_dir=PROJECT_DIR / "models/Qwen2",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        evaluation_strategy="epoch",
        optim="paged_adamw_32bit",
        logging_steps=10,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        bf16=True,
        logging_dir=PROJECT_DIR / "logs/Qwen2",
    )

    # Make the trainer
    metric_fct = partial(
        compute_metrics,
        target_class="essential"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        peft_config=peft_config,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=metric_fct
    )

    return trainer


def do_train(
    model_path: str,
    data_path: str,
    device: DeviceType = 'distributed',
):
    # Build the model / tokenizer
    model, tokenizer, peft_config = _build_model(model_path, device)
    
    # Build the dataset
    dataset_train, dataset_val = _make_datasets(data_path)

    # Make the trainer
    trainer = _setup_trainer(
        model,
        tokenizer,
        peft_config,
        dataset_train,
        dataset_val,
    )

    # Launch training
    trainer.train()


if __name__ == "__main__":
    do_train(
        model_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        data_path=PROJECT_DIR / "data" / "1.1" / "dev",
        device="distributed"
    )