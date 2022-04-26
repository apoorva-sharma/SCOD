import sys
import os

import numpy as np
import glob
import torch

import matplotlib 
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
import matplotlib.pyplot as plt

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("exp_folder", type=str, help="Folder in which to look for config.py to setup experiment")
args = parser.parse_args()


sys.path.append(os.path.abspath(args.exp_folder))
import config

results_dict = {}
folder_path = os.path.join(args.exp_folder, "results","*")
for filename in glob.glob(folder_path):
    name = os.path.basename(filename)
    results_dict[name] = torch.load(filename)

for filename, info in config.plots_to_generate.items():
    summarized_results = info['summary_fn'](results_dict, *info['summary_fn_args'], **info['summary_fn_kwargs'])
    info['plot_fn'](summarized_results, *info['plot_fn_args'], **info['plot_fn_kwargs'])
    if 'legend' in info:
        plt.legend(**info['legend'])
    if 'title' in info:
        plt.title(info['title'])
    plt.tight_layout()
    plt.savefig(os.path.join(args.exp_folder,filename))
    plt.show()