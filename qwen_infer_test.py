import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time


def load_model(model_name_or_path="Qwen/Qwen2-7B-Instruct"):
    """
    Load the Qwen model and tokenizer

    Args:
        model_name_or_path: Model identifier or local path

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_length=2048):
    """
    Generate a response from the model

    Args:
        model: The loaded Qwen model
        tokenizer: The loaded Qwen tokenizer
        prompt: The input prompt
        max_length: Maximum length of generated text

    Returns:
        The generated response
    """
    # Format the prompt according to Qwen's chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize the prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Time the generation
    start_time = time.time()

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Measure elapsed time
    elapsed_time = time.time() - start_time

    # Decode the output
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"Generation time: {elapsed_time:.2f} seconds")
    return response


def generate_response(model, tokenizer, prompt, max_length=2048):
    """
    Generate a response from the model

    Args:
        model: The loaded Qwen model
        tokenizer: The loaded Qwen tokenizer
        prompt: The input prompt
        max_length: Maximum length of generated text

    Returns:
        The generated response
    """
    # Format the prompt according to Qwen's chat template
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)

    # Tokenize the prompt
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Time the generation
    start_time = time.time()

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Measure elapsed time
    elapsed_time = time.time() - start_time

    # Decode the output
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"Generation time: {elapsed_time:.2f} seconds")
    return response

if __name__ == "__main__":
    main()