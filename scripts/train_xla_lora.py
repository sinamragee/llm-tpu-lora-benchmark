import json
import argparse
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from preprocess_data import prepare_dataset

def main():
    parser = argparse.ArgumentParser(description="Distributed LoRA Fine-Tuning on Cloud TPUs")
    parser.add_argument("--config", type=str, default="configs/lora_30b_config.json")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--output_dir", type=str, default="./outputs/gemma4-31b-lora")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    device = xm.xla_device()
    print(f"Initialized XLA device: {device}")

    tokenized_ds = prepare_dataset(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        max_length=cfg["max_seq_length"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_cfg = LoraConfig(
        r=cfg["lora_config"]["r"],
        lora_alpha=cfg["lora_config"]["lora_alpha"],
        lora_dropout=cfg["lora_config"]["lora_dropout"],
        target_modules=cfg["lora_config"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=cfg["xla_settings"]["per_device_batch_size"],
        gradient_accumulation_steps=cfg["xla_settings"]["gradient_accumulation_steps"],
        learning_rate=2e-5,
        num_train_epochs=1,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        gradient_checkpointing=cfg["xla_settings"]["gradient_checkpointing"],
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds
    )

    trainer.train()
    if xm.is_master_ordinal():
        model.save_pretrained(args.output_dir)
        print(f"Training completed. Adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()
