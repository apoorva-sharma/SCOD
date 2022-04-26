import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from nn_ood.data.cifar10 import Cifar10Data
from nn_ood.posteriors import LocalEnsemble, SCOD, Ensemble, Naive, KFAC
from scod.distributions import Categorical
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from densenet import densenet121
    
# WHERE TO SAVE THE MODEL
FILENAME = "model"

## HYPERPARAMS
N_MODELS = 1

LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.9

LR_DROP_FACTOR = 0.5
EPOCHS_PER_DROP = 2

BATCH_SIZE = 5

N_EPOCHS = 0

## SET UP DATASETS
dataset_class = Cifar10Data
test_dataset_args = ['val', 'ood', 'svhn', 'tIN', 'lsun', 'cifar100']

## DEFINE VISUALIZATION FUNCTIONS
def plt_image(ax, inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2471, 0.2435, 0.2616])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return ax.imshow(inp)
    
def viz_dataset_sample(ax, dataset, idx=0, model=None, unc_model=None):
    input, target = dataset[idx]
    plt_image(ax, input)
    xlabel = 'Target: %d' % target
    if unc_model is not None:
        input = input.to(device)
        pred, unc = unc_model(input.unsqueeze(0))
        pred = pred[0].detach().cpu().numpy()
        unc = unc.item()
        xlabel += '\nPred: %d\nUnc: %0.3f' % (np.argmax(pred), unc)
    elif model is not None:
        input = input.to(device)
        pred = model(input.unsqueeze(0))[0].detach().cpu().numpy()
        xlabel += '\nPred: %d' % np.argmax(pred)
        
    ax.set_xlabel(xlabel)


def viz_datasets(idx=0, unc_model=None, model=None):
    num_plots = len(test_dataset_args)
    fig, axes = plt.subplots(1,num_plots, figsize=[5*num_plots, 5], dpi=100)
    for i, split in enumerate( test_dataset_args ):
        dataset = dataset_class(split)
        viz_dataset_sample(axes[i], dataset, idx=idx, unc_model=unc_model, model=model)

## USE CUDA IF POSSIBLE
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classesToKeep = [0,1,2,3,4]

class classSelector(nn.Module):
    def __init__(self, idx_list):
        super().__init__()
        self.idx_list = idx_list
        
    def forward(self,x):
        result = x[..., self.idx_list]
        return result

## MODEL SET UP
og_req_grads = []
def make_model():
    global og_req_grads
    base = densenet121(pretrained=True)
    selector = classSelector(classesToKeep)
    model = nn.Sequential(base, selector)
    
    og_req_grads = [p.requires_grad for p in model.parameters()]
    
    return model

def freeze_model(model, freeze_frac=None):
    # freeze up the the last 
    n_params = len(list(model.parameters()))
    if freeze_frac is None:
        freeze_frac = 1.
    print(freeze_frac)
        
    for i, p in enumerate(model.parameters()):
        if i < freeze_frac*n_params:
            p.requires_grad = False
    


    # make last layer trainable
    for i,m in enumerate(model.children()):
        if i > 0:
            break
        for p in m.classifier.parameters():
            p.requires_grad = True
        
def unfreeze_model(model):
    global og_req_grads
    # unfreeze everything
    for p,v in zip( model.parameters(), og_req_grads):
        p.requires_grad = v

def disable_batchnorm(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        for n,p in m.named_parameters():
            p.requires_grad_(False)
            
def get_features_and_classifier(model):
    """works for Densenet with class selector only"""
    features = nn.Sequential(
        model[0].features,
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    classifier = nn.Sequential(
        model[0].classifier,
        model[1]
    )
    return features, classifier
            
## OPTIMIZATION 
dist_constructor = lambda z: Categorical(logits=z)
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

# recipe for preparing uncertainty models
prep_unc_models = {
    # 'scod_SRFT_s184_n30_freeze_0.85': {
    #     'class': SCOD,
    #     'kwargs': {
    #         'num_samples': 184,
    #         'num_eigs': 30,
    #         'device':'gpu',
    #         'sketch_type': 'srft'
    #     },
    #     'freeze': 0.85
    # },
    'scod_SRFT_s76_n12': {
        'class': SCOD,
        'kwargs': {
            'num_samples': 76,
            'num_eigs': 12,
            'device':'gpu',
            'sketch_type': 'srft'
        },
    },
}


import seaborn as sns

keys_to_compare = []
colors = []

test_unc_models = {}

Meps_to_test = np.exp( np.linspace(2,5,3) )
color_palette = sns.color_palette("crest", len(Meps_to_test))
for i,Meps in enumerate(Meps_to_test):
    exp_name = 'Meps=%0.3e' % (Meps)
    keys_to_compare.append(exp_name)
    exp = {
        'class': SCOD,
        'kwargs': {
            'num_samples': 76,
            'num_eigs': 12,
            'device':'gpu'
        },
        'load_name': 'scod_SRFT_s76_n12',
        'forward_kwargs': {
           'n_eigs': 12,
           'prior_multiplier': Meps,
        }
    }
    
    colors.append(color_palette[i])

    test_unc_models[exp_name] = exp

# OOD PERFORMANCE TESTS
splits_to_use = test_dataset_args
err_thresh = 1.

in_dist_splits = test_dataset_args[0:1]
out_dist_splits = test_dataset_args[2:]

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
    idx = np.argsort(Mepss)
    aurocs = np.array(aurocs)
    auroc_confs = np.array(auroc_confs)
    
    plt.figure(figsize=[4,2.5],dpi=150)
    plt.errorbar(Mepss[idx], aurocs[idx], yerr=auroc_confs[idx],
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
            # 'keys_to_compare': keys_to_compare,
        },
        'plot_fn': plot_ablation,
        'plot_fn_args': [],
        'plot_fn_kwargs': {},
    },
}