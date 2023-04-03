from yaml import load, dump, Loader
import os
from logger import logger

class Config(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
    
    @logger.catch
    def load_config(self):
        # check if config file exists
        if not os.path.exists(self.config_path):
            raise FileNotFoundError('Config file not found')
        # check extension
        if not self.config_path.endswith('.yaml'):
            raise ValueError('Config file must be a yaml file')
        with open(self.config_path, 'r') as f:
            return load(f, Loader=Loader)
    
    @logger.catch
    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        elif '.' in key:
            keys = key.split('.')
            value = self.config
            for k in keys:
                if k in value:
                    value = value[k]
                else:
                    raise KeyError(f'Key {k} not found in config')
            return value
