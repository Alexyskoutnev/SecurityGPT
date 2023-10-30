import os
import yaml

CONFIG_DIR = "./securityGPT/config"

class Cfgloader(object):

    def __init__(self, config_dir : str = CONFIG_DIR) -> None:
        self.config_dir = config_dir

    def load_config(self, config_name : str) -> dict:
        config_path = os.path.join(self.config_dir, config_name)
        with open(config_path, 'r') as cfg_file:
            config = yaml.safe_load(cfg_file)
        return config