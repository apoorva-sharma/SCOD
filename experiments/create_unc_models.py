import sys
import os
import cProfile
import torch

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("exp_folder", type=str, help="Folder in which to look for config.py to setup experiment")
parser.add_argument("--unc_models", type=str, nargs='*', help="Which unc_models to create, coresponding to keys in the config.prep_unc_models. Uses all keys if nothing is passed")
args = parser.parse_args()

sys.path.append(os.path.abspath(args.exp_folder))
import config

save_folder = os.path.join(args.exp_folder, 'times')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
## SET UP MODEL
model = config.make_model()

## LOAD MODEL
filename = os.path.join(args.exp_folder, "models", config.FILENAME + "_0" )
model.load_state_dict(torch.load(filename))
model = model.to(config.device)
model.eval()

## SETUP DATASET
dataset = config.dataset_class("train", N=5000)

if args.unc_models:
    for name in args.unc_models:
        if name not in config.prep_unc_models:
            print(f"{name} is not a valid key in config.prep_unc_models, must be one of {config.prep_unc_models.keys()}")

## SET UP UNC WRAPPERS
for name, info in config.prep_unc_models.items():
    if args.unc_models and name not in args.unc_models:
        continue
    
    print(name)
    config.unfreeze_model(model)
    if 'freeze' in info:
        if type(info['freeze']) == bool:
            freeze_frac = None
        else:
            freeze_frac = info['freeze']
        config.freeze_model(model, freeze_frac=freeze_frac)        
    
    if 'apply_fn' in info:
        model.apply(info['apply_fn'])

    unc_model = info['class'](model, config.dist_constructor, info['kwargs'])

    cProfile.run("""\n
unc_model.process_dataset(dataset)
    """, os.path.join(args.exp_folder, "times", name+"_process.timing") )

    filename = os.path.join(args.exp_folder, "models", name+"_"+config.FILENAME)
    torch.save(unc_model.state_dict(), filename)