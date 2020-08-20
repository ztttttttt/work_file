import logging
import os
import yaml
import json

config = None


def setup_config(default_config_path='config.yaml'):
    global config
    if config:
        return

    # load config file
    config_path = default_config_path
    if not os.path.exists(config_path):
        logging.warning("Config file not found. file path: %s", config_path)
        return

    with open(config_path, 'rt') as f:
        config = yaml.safe_load(f.read())






