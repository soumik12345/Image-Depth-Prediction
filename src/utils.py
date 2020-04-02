import json


def parse_config(json_file):
    with open(json_file, 'r') as f:
        configs = json.load(f)
    return configs
