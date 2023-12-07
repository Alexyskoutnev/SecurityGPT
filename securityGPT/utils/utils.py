import torch
import torch.nn as nn
from datetime import datetime
import os
import yaml

CONFIG_DIR = "./securityGPT/config"
SAVE_DIR = "./models/"

def save_model(model, type=""):
    time = str(datetime.now())
    path = os.path.join(SAVE_DIR, type, time + "_" +  type + ".pth")
    torch.save(model.state_dict(), path)

class Cfgloader(object):

    def __init__(self, config_dir : str = CONFIG_DIR) -> None:
        self.config_dir = config_dir

    def load_config(self, config_name : str) -> dict:
        config_path = os.path.join(self.config_dir, config_name)
        with open(config_path, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
        return config