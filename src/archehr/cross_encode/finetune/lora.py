from peft import get_peft_model, LoraConfig, TaskType

from archehr.utils.loaders import load_model_hf, DeviceType


def do_train(
    model_path: str,
    device: DeviceType = 'distributed',
):

    # Load the model
    model, tokenizer = load_model_hf(
        model_name=model_path,
        device=device
    )

    # Make the PEFT config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, #Sequence cls or Token cls ?
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    
    # Adapt the model for peft
    model = get_peft_model(model, peft_config)
    print(model.print_trainable_parameters())

    print(model)


if __name__ == "__main__":
    do_train(
        model_path="Alibaba-NLP/gte-Qwen2-7B-instruct",
        device="distributed"
    )