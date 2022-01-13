import io
import os
import yaml
import re

from unicode_utils import normalize_text

def _load_config(config_file='./config.yaml'):
    """
    Load the json configuration 
    """

    config = None
    if config_file and os.path.exists(config_file) and os.path.isfile(config_file):
        with open(config_file, 'r') as the_file:
            raw_configuration = the_file.read()
        try:
            configuration = yaml.safe_load(raw_configuration)
        except:
            # note: it appears complicated to get parse error details from the exception
            configuration = None

        if configuration == None:
            msg = "Error: yaml config file cannot be parsed: " + str(config_file)
            raise Exception(msg)
    else:
        msg = "Error: configuration file is not valid: " + str(config_file)
        raise Exception(msg)

    return configuration

def _normalize_text_for_scoring(text): 
    if text == None or len(text) == 0:
        return text
    text = normalize_text(text)
    text = re.sub(r'([ \t\n\r]+)', ' ', text)
    return text.strip()

def _remove_outliner(x, y, ratio=0.05):
    # remove outliner
    x_max = None
    x_min = None
    for i in range(len(x)):
        if x_max == None or x[i][0] > x_max:
            x_max = x[i][0]
        if x_min == None or x[i][0] < x_min:
            x_min = x[i][0]

    interval = x_max - x_min
    internal_ratio = ratio * interval
    x_max = x_max - internal_ratio
    x_min = x_min + internal_ratio

    to_remove = []
    for i in range(len(x)):
        if x[i][0] >= x_max or x[i][0] <= x_min:
            to_remove.insert(0, i)

    for i in to_remove:
        del x[i]
        del y[i]

    return x, y
