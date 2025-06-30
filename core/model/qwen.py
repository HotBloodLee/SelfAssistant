import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from core.model.base_model import SABaseModel


class QwenModel(SABaseModel):
    def __init__(self, config):
        super(QwenModel, self).__init__(config)
        self.p_at_1 = 0
        self.hit_at_5 = 0
        self.mrr = 0
        self.model, self.tokenizer = self.load_model(self.config['model_name_or_path'])


    def load_model(self, model_name_or_path="Qwen/qwen3-0.6b-base"):
        """
            Load the Qwen model and tokenizer

            Args:
                model_name_or_path: Model identifier or local path

            Returns:
                model, tokenizer
            """
        self.logger.info(f"Loading model: {model_name_or_path}")
        tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path,use_cache=False)

        model_ = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map="auto",
            torch_dtype=torch.float16,
            use_cache=False
        )
        return model_, tokenizer_

    def final_metrics(self, count):
        print(f"Precision: `{self.p_at_1/count}`")
        print(f"Hit@5: `{self.hit_at_5/count}`")
        print(f"MRR: `{self.mrr/count}`")

    def inference_per_question(self, rewritten_question, context):
        return self.generate_output(rewritten_question, context)

    def inference(self, data, output_path):
        pass

    def prepare_sft_data(self, data):
        data_list = []
        for d in data:
            question = d["question"]
            context = d["context"]
            answer = d["answer"]

            full_prompt = self.create_prompt(question, context, answer)
            full_prompt = self.tokenizer.apply_chat_template(full_prompt, tokenize=False)

            prompt = self.create_prompt(question, context, None)
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

            entry = dict()
            entry["prompt"] = prompt
            entry["full_prompt"] = full_prompt
            data_list.append(entry)

        return data_list

    def prepare_dpo_data(self, data):
        data_list = []
        for d in data:
            question = d["question"]
            context = d["context"]
            answer = d["answer"]
            non_preferred = d["non_preferred"]
            sample = dict()

            prompt = self.create_prompt(question, context)
            prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False)

            sample["prompt"] = prompt
            sample["chosen"] = answer
            sample["rejected"] = non_preferred

            data_list.append(sample)

        return data_list