import yaml

def load_params():
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)