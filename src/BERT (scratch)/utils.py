import os
from src.globals import *

def read_samples(type=None):
    if not type:
        raise Exception("Sample type is not detected! You need to specify type to read sample texts")
    for filename in os.listdir(os.path.join(DIR_DATA,type)):
        if filename.endswith(".txt"):
            with open(os.path.join(DIR_DATA,type,filename)) as f:
                sample = f.read()
        yield filename.split('.')[0],sample

def get_trigger_templates():
    #– “what is the trigger”, “trigger”, “action”, “verb”
    return ['trigger']
    #return ['what is the trigger', 'trigger', 'action', 'verb']

def get_argument_templates(type=None):
    """
    template 1: what is the [arg]
    template 2: what is the [arg] in [event trigger]
    template 3: description
    """
    if not type:
        raise Exception("Sample type is not detected! You need to specify type to use template questions")
    templates = {}
    with open(type+"_templates.csv", "r") as f:
        for line in f:
            type,template = line.strip().split(",")
            templates[type.split('_')[-1]] = template
    return templates