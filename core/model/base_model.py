class SABaseModel():

    def __init__(self):
        pass

    def set_model(self):
        pass

    def set_generation_pipeline(self):
        pass

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
        pass

    def add_assistant_ouput(self, answer):
        pass

    def create_prompt(self, question, context, label=None):
        pass

    def generate_output(self, question: str, context: dict = {}, create_prompt=None):
        pass

    def return_prompt_and_responses(self, samples):
        pass

    def train(self, train_in_path, train_out_path, eval_in_path=None, eval_out_path=None):
        pass

    def train_dpo(self, train_in_path, train_out_path, eval_in_path=None, eval_out_path=None):
        pass

