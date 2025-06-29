import copy
import random
from pprint import pprint

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM
from trl import  DPOTrainer, DPOConfig

from core.utils.common_utils import LossRecorderCallback, plot_training_loss

IGNORE_INDEX = -100

def load_model(model_name_or_path="Qwen/qwen3-0.6b-0.6B"):
    """
    Load the Qwen model and tokenizer

    Args:
        model_name_or_path: Model identifier or local path

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_name_or_path}")
    tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path, use_cache=False)
    model_ = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False
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
        "answer": "Paris",
        "non_preferred": "Unknown"
    },
    {
        "question": "Who wrote '1984'?",
        "context": "George Orwell was a British writer.",
        "answer": "George Orwell",
        "non_preferred": "Not sure"
    },
    {
        "question": "What is the tallest mountain in the world?",
        "context": "Mountains are natural elevations of the Earth's surface.",
        "answer": "Mount Everest",
        "non_preferred": "K2"
    },
    {
        "question": "What is the boiling point of water?",
        "context": "Water boils at a specific temperature under normal atmospheric pressure.",
        "answer": "373.15°C",
        "non_preferred": "100°C"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "context": "The largest planet in our solar system is Jupiter.",
        "answer": "Jupiter",
        "non_preferred": "Saturn"
    },
    {
        "question": "What is the chemical symbol for gold?",
        "context": "Gold is a precious metal used in jewelry and electronics.",
        "answer": "Au",
        "non_preferred": "Ag"
    },
    {
        "question": "What is the chemical symbol for water?",
        "context": "Water is the chemical element with the symbol H2O.",
        "answer": "H2O",
        "non_preferred": "HO2"
    },
    {
        "question": "Who discovered penicillin?",
        "context": "Penicillin is an antibiotic that revolutionized medicine.",
        "answer": "Alexander Fleming",
        "non_preferred": "Louis Pasteur"
    },
    {
        "question": "What is the currency of Japan?",
        "context": "Japan is an island nation in East Asia.",
        "answer": "Yen",
        "non_preferred": "Won"
    },
    {
        "question": "What is the largest ocean in the world?",
        "context": "The largest ocean in the world is the Pacific Ocean.",
        "answer": "Pacific Ocean",
        "non_preferred": "Atlantic Ocean"
    }
]

model, tokenizer = load_model("model/qwen3-0.6b-base")

data_list = []
for d in data:
    question = d["question"]
    context = d["context"]
    answer = d["answer"]
    non_preferred = d["non_preferred"]
    sample = dict()

    prompt = create_prompt(question, context)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    sample["prompt"] = prompt
    sample["chosen"] = answer
    sample["rejected"] = non_preferred

    data_list.append(sample)


pprint(data_list)


random.shuffle(data_list)
train_list = data_list[:int(len(data_list) * 0.8)]
dev_list= data_list[int(len(data_list) * 0.8):]

train_dataset = Dataset.from_list(train_list)

# dev_dataset = None
dev_dataset = Dataset.from_list(dev_list)


print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Available GPUs: {torch.cuda.device_count()}")

if dev_dataset is None:
    evalDuringTraining = "no"
else:
    evalDuringTraining = "epoch"

training_args = DPOConfig(
    output_dir="./out",
    learning_rate=0.000001,#1e-6,
    warmup_steps=150,#150,
    per_device_train_batch_size=1, #2
    per_device_eval_batch_size=1,
    num_train_epochs=20,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    eval_strategy=evalDuringTraining,
    save_strategy="epoch",
    save_only_model=True,
    save_total_limit=1,
    bf16=torch.cuda.is_bf16_supported(),
    beta=0.1
)

loss_recorder = LossRecorderCallback()

optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, eps=1e-4)

# trainer
dpo_trainer = DPOTrainer(
    model,                  # base model from SFT pipeline
    ref_model=None,              # typically a copy of the SFT trained base model
    train_dataset=train_dataset, # dataset prepared above
    eval_dataset=dev_dataset,           # eval dataset prepared above
    tokenizer=tokenizer,    # tokenizer
    args=training_args,          # training arguments e.g. batch size, lr, etc.
    optimizers=(optimizer, None),  # No scheduler
    callbacks=[loss_recorder],
)

print("Starting training now...")
dpo_trainer.train()
plot_training_loss(loss_recorder, save_path="./dpo_loss_plot.png")
print("Done with training!")