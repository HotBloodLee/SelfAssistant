import copy
import json
import logging
import os
import random

import torch
from datasets import Dataset
from transformers import pipeline, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from trl import DPOTrainer, DPOConfig

from core.utils.common_utils import IGNORE_INDEX, LossRecorderCallback, plot_training_loss


class SABaseModel():
    def __init__(self, config):
        self.config = config
        self.comp_id = config['comp_id']
        self.set_logger(config)

        self.instruction = config['instruction']
        self.chat_prompt_template = config['chat_prompt_template']

        self.tokenizer = None

    def set_logger(self, config):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        file_handler = logging.FileHandler(config['log_path'], mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def set_model(self, model):
        self.model = model

    def set_generation_pipeline(self):
        self.generation_pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=75,
        )

    def load_model(self):
        pass

    def prepare_sft_data(self, data):
        pass


    def prepare_pref_data(self, data, ev_data=None):
        pass


    def calculate_metrics(self, input1, input2):
        pass


    def final_metrics(self, count):
        pass

    def load_data(self, path):
        if not path or not path.endswith((".json", ".jsonl")) or not os.path.exists(path):
            raise ValueError("Path must be a valid JSON or JSONL file.")
        if path.endswith(".jsonl"):
            with open(path, "r") as efp:
                data = []
                line = efp.readline()

                while line:
                    conversation = json.loads(line)
                    data.append(conversation)
                    line = efp.readline()
        else:
            with open(path, "r") as efp:
                data = json.load(efp)
        return data

    def add_assistant_output(self, answer):
        return [{
            "role": "assistant",
            "content": self.comp_id + ": " + answer,
        }]

    def create_prompt(self, question, context, label=None):
        dialog = [
            {
                "role": "system",
                "content": self.instruction
            },
            {
                "role": "user",
                "content": self.chat_prompt_template.format(question=question, context=context),
            }
        ]

        if not label is None:
            if self.comp_id != "":
                labelString = self.comp_id + ": " + label
            else:
                labelString = label
            dialog.append({
                "role": "assistant",
                "content": labelString
            })
        return dialog

    def generate_output(self, question: str, context: dict = {}, create_prompt=None):
        pass

    def return_prompt_and_responses(self, samples):
        if "model_max_length" in self.config.keys():
            input_encodings = self.tokenizer(samples["full_prompt"], truncation=True, padding='max_length',
                                             max_length=self.config["model_max_length"])
            model_prompts = self.tokenizer(samples["prompt"], truncation=True, padding='max_length',
                                           max_length=self.config["model_max_length"])
        else:
            input_encodings = self.tokenizer(samples["full_prompt"])
            model_prompts = self.tokenizer(samples["prompt"])

        label = copy.deepcopy(input_encodings["input_ids"])
        prompt_length = len(model_prompts["input_ids"])
        # 0是起始标记，不能mask
        label[1:prompt_length - 1] = [IGNORE_INDEX] * (prompt_length - 2)  # mask the query

        new_label = [(l if l != self.tokenizer.pad_token_id else IGNORE_INDEX) for l in label]
        input_encodings["labels"] = new_label

        return input_encodings

    def train(self, train_list, dev_list=None):
        random.shuffle(train_list)
        train_dataset = Dataset.from_list(train_list)

        train_dataset = train_dataset.map(
            self.return_prompt_and_responses,
            batched=False,
        )
        train_dataset = train_dataset.remove_columns(['prompt', 'full_prompt'])

        for index in random.sample(range(len(train_dataset)), 1):
            self.logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        dev_dataset = None
        if not dev_list is None:
            random.shuffle(dev_list)
            dev_dataset = Dataset.from_list(dev_list)

            dev_dataset = dev_dataset.map(
                self.return_prompt_and_responses,
                batched=False,
            )
            dev_dataset = dev_dataset.remove_columns(['prompt', 'full_prompt'])

        self.logger.info(f"Cuda available: {torch.cuda.is_available()}")
        self.logger.info(f"Available GPUs: {torch.cuda.device_count()}")

        if dev_dataset is None:
            evalDuringTraining = "no"
        else:
            evalDuringTraining = "epoch"

        training_args = TrainingArguments(
            output_dir=self.config["model_save_path"],
            warmup_ratio=self.config["model_warmup_ratio"],
            learning_rate=self.config["model_learningrate"],
            num_train_epochs=self.config["model_num_epochs"],
            per_device_train_batch_size=self.config["model_batch_size"],
            per_device_eval_batch_size=self.config["model_eval_batch_size"],
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

        )

        loss_recorder = LossRecorderCallback()

        # optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_args.learning_rate, eps=1e-4)

        # trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer, return_tensors="pt"),
            optimizers=(optimizer, None),  # No scheduler
            callbacks=[loss_recorder]
        )

        self.logger.info("Starting training now...")
        trainer.train()
        plot_training_loss(loss_recorder, save_path="./sft_loss.png")
        self.logger.info("Done with training!")

    def train_dpo(self, train_list, dev_list=None):
        self.logger.info(f"Cuda available: {torch.cuda.is_available()}")
        self.logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        train_dataset = Dataset.from_list(train_list)
        del train_list

        dev_dataset = None
        if not dev_list is None:
            dev_dataset = Dataset.from_list(dev_list)
            del dev_list

        evalDuringTraining = True
        if dev_dataset is None:
            evalDuringTraining = "no"
        else:
            evalDuringTraining = "epoch"

        training_args = DPOConfig(
            output_dir=self.config["model_save_path"],
            learning_rate=self.config["model_learningrate"],  # 1e-6,
            warmup_steps=self.config["model_warmup_steps"],  # 150,
            per_device_train_batch_size=self.config["model_batch_size"],  # 2
            per_device_eval_batch_size=self.config["model_eval_batch_size"],
            num_train_epochs=self.config["model_num_epochs"],
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            eval_strategy=evalDuringTraining,
            save_strategy="epoch",
            save_only_model=True,
            bf16=torch.cuda.is_bf16_supported(),
            beta=0.1
        )

        loss_recorder = LossRecorderCallback()

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_args.learning_rate, eps=1e-4)

        dpo_trainer = DPOTrainer(
            self.model,  # base model from SFT pipeline
            ref_model=None,  # typically a copy of the SFT trained base model
            train_dataset=train_dataset,  # dataset prepared above
            eval_dataset=None,  # eval dataset prepared above
            tokenizer=self.tokenizer,  # tokenizer
            args=training_args,  # training arguments e.g. batch size, lr, etc.
            optimizers=(optimizer, None),  # No scheduler
            callbacks=[loss_recorder],
        )

        self.logger.info("Starting training now...")
        dpo_trainer.train()
        plot_training_loss(loss_recorder, save_path="./dpo_loss.png")
        self.logger.info("Done with training!")

