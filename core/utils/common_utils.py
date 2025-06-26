import yaml

IGNORE_INDEX = -100

def load_config(config_path):
    """Load config from yaml path."""
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    return config