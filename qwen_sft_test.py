import copy
import random
from pprint import pprint

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM

IGNORE_INDEX = 151643

def load_model(model_name_or_path="Qwen/qwen3-0.6b-0.6B"):
    """
    Load the Qwen model and tokenizer

    Args:
        model_name_or_path: Model identifier or local path

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path)
    print(tokenizer_.pad_token_id)
    print(tokenizer_.vocab_size)
    model_ = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16
    )
    return model_, tokenizer_

def return_prompt_and_responses(samples):
    input_encodings = tokenizer(samples["full_prompt"])
    model_prompts = tokenizer(samples["prompt"])


    label = copy.deepcopy(input_encodings["input_ids"])
    prompt_length = len(model_prompts["input_ids"])
    label[1:prompt_length - 1] = [IGNORE_INDEX] * (prompt_length - 2)  # mask the query

    new_label = [(l if l != tokenizer.pad_token_id else IGNORE_INDEX) for l in label]
    input_encodings["labels"] = new_label

    return input_encodings

def create_prompt(question, context, label=None):
    dialog = [
        {
            "role": "system",
            "content": (
                "You are given a question and references which may or may not help answer the question.  "
                "Please answer the question in as few words as possible by using the information provided in the references that is relevant in answering the question. "
            )
        },
    ]
    CHAT_PROMPT_TEMPLATE = (
        "question: {question}\n"
        "context: {context}\n"
    )

    dialog.append({
        "role": "user",
        "content": CHAT_PROMPT_TEMPLATE.format(question=question, context=context),
    })

    if not label is None:
        labelString = "answer: " + label
        dialog.append({
            "role": "assistant",
            "content": labelString
        })
    return dialog

data = [
    {
        "question": "What is the capital of France?",
        "context": "France is a country in Europe.",
        "answer": "Paris"
    },
    {
        "question": "Who wrote '1984'?",
        "context": "George Orwell was a British writer.",
        "answer": "George Orwell"
    },
    {
        "question": "What is the tallest mountain in the world?",
        "context": "Mountains are natural elevations of the Earth's surface.",
        "answer": "Mount Everest"
    },
    {
        "question": "What is the boiling point of water?",
        "context": "Water boils at a specific temperature under normal atmospheric pressure.",
        "answer": "373.15°C"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "context": "The largest planet in our solar system is Jupiter.",
        "answer": "Jupiter"
    },
    {
        "question": "What is the chemical symbol for gold?",
        "context": "Gold is a precious metal used in jewelry and electronics.",
        "answer": "Au"
    },
    {
        "question": "What is the chemical symbol for water?",
        "context": "Water is the chemical element with the symbol H2O.",
        "answer": "H2O"
    },
    {
        "question": "Who discovered penicillin?",
        "context": "Penicillin is an antibiotic that revolutionized medicine.",
        "answer": "Alexander Fleming"
    },
    {
        "question": "What is the currency of Japan?",
        "context": "Japan is an island nation in East Asia.",
        "answer": "Yen"
    },
    {
        "question": "What is the largest ocean in the world?",
        "context": "The largest ocean in the world is the Pacific Ocean.",
        "answer": "Pacific Ocean"
    }
]

model, tokenizer = load_model("model/qwen3-0.6b-base")

data_list = []
for d in data:
    question = d["question"]
    context = d["context"]
    answer = d["answer"]

    full_prompt = create_prompt(question, context, answer)
    full_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False)

    prompt = create_prompt(question, context, None)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    entry = dict()
    entry["prompt"] = prompt
    entry["full_prompt"] = full_prompt
    data_list.append(entry)


pprint(data_list)


random.shuffle(data_list)
train_list = data_list[:int(len(data_list) * 0.8)]
dev_list= data_list[int(len(data_list) * 0.8):]

train_dataset = Dataset.from_list(train_list)
train_dataset = train_dataset.map(
    return_prompt_and_responses,
    batched=False,
)
train_dataset = train_dataset.remove_columns(['prompt', 'full_prompt'])
# dev_dataset = None
dev_dataset = Dataset.from_list(dev_list)
dev_dataset = dev_dataset.map(
    return_prompt_and_responses,
    batched=False,
)
dev_dataset = dev_dataset.remove_columns(['prompt', 'full_prompt'])

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Available GPUs: {torch.cuda.device_count()}")

if dev_dataset is None:
    evalDuringTraining = "no"
else:
    evalDuringTraining = "epoch"

training_args = TrainingArguments(
    output_dir="./out",
    warmup_ratio=0.01,
    learning_rate=0.000001,
    num_train_epochs=2,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,  # prevents from indexing errors
    save_strategy="epoch",  # epoch-wise eval
    eval_strategy=evalDuringTraining,  # epoch-wise eval
    save_only_model=True,  # do not store optimizer state etc. to save space
    save_total_limit=1,  # only store best model
    report_to="none",  # avoid issues with distributed setup
    # bf16=torch.cuda.is_bf16_supported(),  # mixed-precision training
    bf16=False,
    # do_eval=evalDuringTraining,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,

)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

print("Starting training now...")
trainer.train()
print("Done with training!")