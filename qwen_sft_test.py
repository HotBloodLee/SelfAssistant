import copy
import json
import random
from pprint import pprint

import torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

from core.utils.common_utils import LossRecorderCallback, plot_training_loss, load_json

IGNORE_INDEX = -100

def load_model(model_name_or_path="Qwen/qwen3-0.6b", use_lora=False):
    print(f"Loading model: {model_name_or_path}")
    tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path, use_cache=False)

    # 添加pad_token如果不存在
    if tokenizer_.pad_token is None:
        tokenizer_.pad_token = tokenizer_.eos_token

    model_ = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_cache=False
    )

    if use_lora:
        lora_path = './out/checkpoint-699'

        # 配置LoRA参数
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )

        # 应用LoRA
        model_ = PeftModel.from_pretrained(model_)
        model_.print_trainable_parameters()

    return model_, tokenizer_

def return_prompt_and_responses(samples):
    input_encodings = tokenizer(
        samples["full_prompt"],
        truncation=True,
        max_length=2048,
        padding=False
    )
    model_prompts = tokenizer(
        samples["prompt"],
        truncation=True,
        max_length=2048,
        padding=False
    )

    label = copy.deepcopy(input_encodings["input_ids"])
    prompt_length = len(model_prompts["input_ids"])
    label[1:prompt_length - 1] = [IGNORE_INDEX] * (prompt_length - 2)  # mask the query

    new_label = [(l if l != tokenizer.pad_token_id else IGNORE_INDEX) for l in label]
    input_encodings["labels"] = new_label

    return input_encodings

def create_prompt(input_, label=None):
    dialog = [
        {
            "role": "system",
            "content": (
                "You are an intelligent document parser. Given a document in plain text extracted from a file (PDF, PPT, Excel, Word, or Markdown), your task is to extract its structural content and organize it into a clean, human-readable JSON format. The output should reflect the document's internal organization (such as sections, slides, sheets, etc.) and preserve meaningful information."
                "Please Analyze the input text and convert it into a structured JSON object that faithfully represents the content and hierarchy of the original document. The JSON should include a \"type\" field describing the document type (e.g., \"academic_pdf\", \"ppt_teaching\", \"excel_sheet\", \"markdown_doc\", \"word_handout\"), and then include extracted content fields such as \"sections\", \"chapters\", \"sheets\", etc., as appropriate."
            )
        },
    ]
    CHAT_PROMPT_TEMPLATE = (
        "input: {input}\n"
    )

    dialog.append({
        "role": "user",
        "content": CHAT_PROMPT_TEMPLATE.format(input=input_),
    })

    if not label is None:
        labelString = "output: " + label
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

model, tokenizer = load_model("model/qwen3-0.6b-base", use_lora=True)

data_list = []
for d in data:
    question = d["question"]
    context = d["context"]
    answer = d["answer"]
    non_preferred = d["non_preferred"]

    full_prompt = create_prompt(question+context, answer)
    full_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False)

    prompt = create_prompt(question+context, None)
    prompt = tokenizer.apply_chat_template(prompt, tokenize=False)

    entry = dict()
    entry["prompt"] = prompt
    entry["full_prompt"] = full_prompt
    data_list.append(entry)


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
    learning_rate=0.0002,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    remove_unused_columns=False,  # prevents from indexing errors
    save_strategy="epoch",  # epoch-wise eval
    eval_strategy=evalDuringTraining,  # epoch-wise eval
    save_only_model=True,  # do not store optimizer state etc. to save space
    save_total_limit=1,  # only store best model
    report_to="none",  # avoid issues with distributed setup
    bf16=torch.cuda.is_bf16_supported(),  # mixed-precision training
    do_eval=evalDuringTraining== "epoch",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    dataloader_pin_memory=False,  # 避免内存问题
)

# optimizer

loss_recorder = LossRecorderCallback()

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=training_args.learning_rate, eps=1e-4)

# trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    callbacks=[loss_recorder],
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        pad_to_multiple_of=8
    ),
    optimizers=(optimizer, None),
)

print("Starting training now...")
trainer.train()

# 保存LoRA适配器
model.save_pretrained("./out/lora_adapter")
tokenizer.save_pretrained("./out/lora_adapter")

plot_training_loss(loss_recorder, save_path="./sft_loss.png")
print("Done with training!")