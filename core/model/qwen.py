import sys

from core.model.base_model import SABaseModel


class QwenModel(SABaseModel):
    def __init__(self):
        self.p_at_1 = 0
        self.hit_at_5 = 0
        self.mrr = 0
        super(QwenModel, self).__init__()

    def final_metrics(self, count):
        print(f"Precision: `{self.p_at_1/count}`")
        print(f"Hit@5: `{self.hit_at_5/count}`")
        print(f"MRR: `{self.mrr/count}`")

    def inference_per_question(self, rewritten_question, context):
        return self.generate_output(rewritten_question, context)

    def inference(self, data, output_path):
        pass

    def prepare_sft_data(self, data):
        pass


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]