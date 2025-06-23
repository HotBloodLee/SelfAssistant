import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def download_qwen_model(model_name_or_path="Qwen/Qwen3-0.6b-0.6B", save_directory="./model"):
    """
    Download Qwen model and tokenizer to a specified directory.

    Args:
        model_name_or_path: Model identifier or local path
        save_directory: Directory to save the model and tokenizer
    """
    print(f"Downloading model: {model_name_or_path}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=save_directory
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=save_directory
    )

    print(f"Model and tokenizer saved to: {save_directory}")


if __name__ == "__main__":
    # Specify the save directory
    save_directory = "./model"
    download_qwen_model(save_directory=save_directory)