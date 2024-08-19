import argparse
import os

import yaml
import json
import torch


def get_config():
    # Create the parser
    parser = argparse.ArgumentParser(description="Example script to demonstrate argparse usage.")

    # Define the arguments
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration file.")
    # parser.add_argument("--verbose", action="store_true",
    #                     help="Enable verbose mode.")

    # Parse the arguments
    args = parser.parse_args()

    # Convert argparse Namespace to a dictionary (optional)
    config = vars(args)

    with open(os.path.join("../..", config["config"])) as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    json.dumps(config, sort_keys = True, indent = 4)

    return config


def get_metric_by_name(name: str):
    if name == "cosine":
        return lambda x, y: 1 - torch.nn.functional.cosine_similarity(x, y, dim=0)
    elif name == "euclidean":
        return lambda x, y: torch.norm(x - y, dim=0)
    else:
        raise ValueError
