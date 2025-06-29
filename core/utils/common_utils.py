import math
from pprint import pprint

import yaml
import matplotlib.pyplot as plt
from transformers.trainer_callback import TrainerCallback

IGNORE_INDEX = -100

def load_config(config_path):
    """Load config from yaml path."""
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config


class LossRecorderCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_loss = []
        self.eval_loss = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        # Record training loss
        if "loss" in logs:
            self.training_loss.append(logs["loss"])
            self.steps.append(state.global_step)

        # Record evaluation loss (skipping nan values)
        if "eval_loss" in logs and not math.isnan(logs["eval_loss"]):
            self.eval_loss.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)


def plot_training_loss(callback, save_path="training_loss_plot.png"):
    """Plot and save the training and evaluation loss curves"""
    plt.figure(figsize=(10, 6))

    # Plot training loss
    if callback.training_loss:
        plt.plot(callback.steps, callback.training_loss, label="Training Loss", color="blue")

    # Plot evaluation loss
    if callback.eval_loss:
        plt.plot(callback.eval_steps, callback.eval_loss, label="Evaluation Loss", color="red", marker="o")

    plt.title("Training and Evaluation Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Loss plot saved to {save_path}")