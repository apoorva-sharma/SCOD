import torch
import torch.nn as nn
from nn_ood.data.rotated_mnist import RotatedMNIST
from nn_ood.posteriors import SCOD
from scod.distributions import Normal
import numpy as np
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 1

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 5

BATCH_SIZE = 16

N_EPOCHS = 50

## SET UP DATASETS
dataset_class = RotatedMNIST
test_dataset_args = ['val', 'ood', 'ood_angle']

## Dataset visualization
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.1307
    std = 0.3081
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    ax.imshow(inp[:,:,0], cmap='Greys')
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %0.2f' % target.item()
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred[0].item()
        unc = unc.item()
        xlabel += '\nPred: %02f\nUnc: %0.3f' % (pred, unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0))[0]
        xlabel += '\nPred: %0.3f' % pred.item()
     
    ax.set_xlabel(xlabel)

def viz_datasets(idx=0, unc_model=None, model=None):
    num_plots = len(test_dataset_args)
    fig, axes = plt.subplots(1,num_plots, figsize=[5*num_plots, 5], dpi=100)
    for i, split in enumerate( test_dataset_args ):
        dataset = dataset_class(split)
        viz_dataset_sample(axes[i], dataset, idx=idx, unc_model=unc_model, model=model)
    
## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## MODEL SET UP
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data, gain=5./3)
    if m.__class__.__name__.find('Conv') != -1:
        nn.init.normal_(m.weight, 0.0, 0.02)

def make_model():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, 1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 32, 3, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(288, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    model.apply(weight_init)
    
    return model

def freeze_model(model):
    # freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # make linear layers trainable
    for p in model.children[-2].parameters():
        p.requires_grad = True
    for p in model.children[-1].parameters():
        p.requires_grad = True
    
def unfreeze_model(model):
    # unfreeze everything
    for p in model.parameters():
        p.requires_grad = True

dist_constructor = lambda z: Normal(loc=z, scale=torch.ones_like(z))
opt_class = torch.optim.SGD
opt_kwargs = {
    'lr': LEARNING_RATE,
    'momentum': SGD_MOMENTUM
}
sched_class = torch.optim.lr_scheduler.StepLR
sched_kwargs = {
    'step_size': EPOCHS_PER_DROP,
    'gamma': LR_DROP_FACTOR
}    
    
prep_unc_models = {
    'scod_SRFT_s604_n100': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 604,
            'num_eigs': 100,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
}


import seaborn as sns

keys_to_compare = []
colors = []

test_unc_models = {}

Meps_to_test = np.exp( np.linspace(-8,5,8) )
color_palette = sns.color_palette("crest", len(Meps_to_test))
for i,Meps in enumerate(Meps_to_test):
    exp_name = 'Meps=%0.3e' % (Meps)
    keys_to_compare.append(exp_name)
    exp = {
        'class': SCOD,
        'kwargs': {
            'num_samples': 604,
            'num_eigs': 100,
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s604_n100',
        'forward_kwargs': {
           'n_eigs': 100,
           'prior_multiplier': Meps
        }
    }
    
    colors.append(color_palette[i])

    test_unc_models[exp_name] = exp

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = ['val']
out_dist_splits = ['ood','ood_angle']

# Visualization
from nn_ood.utils.viz import summarize_ood_results
import matplotlib.pyplot as plt

def plot_ablation(summarized_results):
    Mepss = []
    aurocs = []
    auroc_confs = []
    for key,stats in summarized_results.items():
        tokenized = key.replace(',',' ').replace('=',' ').split(' ')
        Meps = float(tokenized[1])
        Mepss.append(Meps)
        aurocs.append(stats['auroc'])
        auroc_confs.append(stats['auroc_conf'])

    Mepss = np.array(Mepss)
    aurocs = np.array(aurocs)
    auroc_confs = np.array(auroc_confs)
    
    plt.figure(figsize=[4,2.5],dpi=150)
    plt.errorbar(Mepss, aurocs, yerr=auroc_confs,
                linestyle='-', marker='', capsize=2)
    plt.xscale('log')
    plt.xlabel(r'Value of $\epsilon^2$')
    plt.ylabel('AUROC')
    
plots_to_generate = {
    'auroc_vs_Meps.pdf': {
        'summary_fn': summarize_ood_results,
        'summary_fn_args': [
            in_dist_splits,
            out_dist_splits
        ],
        'summary_fn_kwargs': {
            'keys_to_compare': keys_to_compare,
        },
        'plot_fn': plot_ablation,
        'plot_fn_args': [],
        'plot_fn_kwargs': {},
    },
}