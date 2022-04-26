import sys
import os
import cProfile
import torch

from nn_ood.utils.train import train_ensemble

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("exp_folder", type=str, help="folder in which to look for config.py to setup experiment")
args = parser.parse_args()


sys.path.append(os.path.abspath(args.exp_folder))
import config



from nn_ood.utils.train import train_ensemble
models = train_ensemble(config.N_MODELS, 
                        config.make_model, 
                        config.dataset_class, 
                        config.dist_constructor, 
                        config.opt_class,
                        config.opt_kwargs,
                        config.sched_class,
                        config.sched_kwargs,
                        config.device,
                        num_epochs=config.N_EPOCHS,
                        batch_size=config.BATCH_SIZE)


## SAVE MODEL
print("saving models")
save_folder = os.path.join(args.exp_folder, 'models')
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for i, model in enumerate(models):
    filename = os.path.join(args.exp_folder, "models", config.FILENAME + "_%d" % i)
    torch.save(model.state_dict(), filename)