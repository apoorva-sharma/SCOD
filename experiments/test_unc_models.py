import sys
import os
import cProfile
import torch

from nn_ood.utils.test import process_datasets

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("exp_folder", type=str, help="Folder in which to look for config.py to setup experiment")
parser.add_argument("--unc_models", type=str, nargs='*', help="Which unc_wrappers to create, coresponding to keys in the config.test_unc_models. Uses all keys if nothing is passed")
args = parser.parse_args()


sys.path.append(os.path.abspath(args.exp_folder))
import config


# LOAD UNC_WRAPPERS
print("Loading models")
models = []
for i in range(config.N_MODELS):
    print("loading model %d" % i)
    filename = os.path.join(args.exp_folder, 'models', config.FILENAME + "_%d" % i)
    state_dict = torch.load(filename)
    model = config.make_model()
    model.load_state_dict(state_dict)
    model.eval()
    model.to(config.device)
    models.append(model)

model = models[0]

save_folder = os.path.join(args.exp_folder, 'results')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_folder = os.path.join(args.exp_folder, 'times')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)


if args.unc_models:
    for name in args.unc_models:
        if name not in config.test_unc_models:
            print(f"{name} is not a valid key in config.test_unc_models, must be one of {config.test_unc_models.keys()}")


exceptions = []

for name, info in config.test_unc_models.items():
    if args.unc_models and name not in args.unc_models:
        continue
    print(name)
    
    config.unfreeze_model(model)
    if 'freeze' in info:
        if type(info['freeze']) is bool:
            freeze_frac = None
        else:
            freeze_frac = info['freeze']
        config.freeze_model(model, freeze_frac=freeze_frac)        
    
    if 'apply_fn' in info:
        model.apply(info['apply_fn'])
        
    if 'multi_model' in info:
        unc_model = info['class'](models, config.dist_constructor, info['kwargs'])
    else:
        unc_model = info['class'](model, config.dist_constructor, info['kwargs'])
    
    if info['load_name'] is not None: 
        filename = os.path.join(args.exp_folder, "models", info['load_name']+"_"+config.FILENAME)
        print(filename)
        unc_model.load_state_dict(torch.load(filename))
        unc_model.cuda()
    
    try:
        cProfile.run("""\n
results = process_datasets(config.dataset_class, 
                           config.test_dataset_args,
                           unc_model, 
                           config.device,
                           N=1000,
                           **info['forward_kwargs'])
        """, os.path.join(args.exp_folder, "times", name) )
        savepath = os.path.join(args.exp_folder, "results", name)
        torch.save(results, savepath)
    except Exception as e:
        exceptions.append(e)

if len(exceptions) > 0:
    print(f"Had {len(exceptions)} failed runs, raising first exception")
    raise exceptions[0]
