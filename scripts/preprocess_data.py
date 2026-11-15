import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def prepare_dataset(model_name: str, dataset_name: str, max_length: int = 2048):
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_ds = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing sequences for XLA static shapes",
    )
    return tokenized_ds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset with static shapes for TPU training")
    parser.add_argument("--model_name", type=str, default="google/gemma-4-31B-it")
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--max_length", type=int, default=2048)
    args = parser.parse_args()

    ds = prepare_dataset(args.model_name, args.dataset_name, args.max_length)
    print(f"Prepared {len(ds)} sequences ready for TPU XLA execution.")
