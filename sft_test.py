import random

from core.model.qwen import QwenModel
from core.utils.common_utils import load_config

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

config = load_config('config/sa_sft.yml')

model = QwenModel(config=None)

data_list = model.prepare_sft_data(data)
random.shuffle(data_list)
train_list = data_list[:int(len(data_list) * 0.8)]
dev_list= data_list[int(len(data_list) * 0.8):]

model.train(train_list, dev_list)