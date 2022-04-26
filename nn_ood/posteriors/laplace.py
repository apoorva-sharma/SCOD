import laplace
from laplace.matrix import Kron
import torch
import torch.nn as nn
from copy import deepcopy
from scod.distributions import Categorical

base_config = {
    'input_shape': [1,28,28],
    'batch_size': 32,
    'device': 'cpu',
    'num_loss_samples': 30,
    'num_samples': 10,
    'damping': False,
}

class KronLaplace(nn.Module):
    """
    Wraps a trained model with functionality for adding epistemic uncertainty estimation.
    
    Only works with models which have output dimension of 1.
    """
    def __init__(self, model, dist_constructor, args={}):
        super().__init__()
        
        self.config = deepcopy(base_config)
        self.config.update(args)
        
        self.model = model
        self.device = next(model.parameters()).device
        self.dist_constructor = dist_constructor
        
        zero_input = torch.zeros([1] + self.config['input_shape'], device=self.device)
        ex_dist = self.dist_constructor(self.model(zero_input))
        self.likelihood = 'regression'
        if type(ex_dist) is Categorical:
            self.likelihood = 'classification'

        self.batch_size = self.config['batch_size']

        self.log_sigma_noise = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
        self.log_prior_sig = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
        self.log_temp = nn.Parameter(torch.zeros(1, device=self.device), requires_grad=True)
        self.hyperparameters = [self.log_sigma_noise, self.log_prior_sig, self.log_temp]

        self.laplace = laplace.KronLaplace(self.model, self.likelihood, 
                                   sigma_noise=self.sigma_noise,
                                   prior_precision=self.prior_prec.detach().cpu(),
                                   temperature=self.temp,
                                   damping=self.config['damping'])

        # initialize, and set up kfacs as nn.parameters so that they are part of the state-dict
        self.factors = nn.ModuleList()
        self.configured = nn.Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=False)

        self.n_data = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=False)
        self.n_outputs = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=False)
        self.loss = nn.Parameter(torch.zeros(1, dtype=torch.float), requires_grad=False)

        self.laplace._init_H()
        for fac in self.laplace.H.kfacs:
            facModule = nn.ParameterList(nn.Parameter(p, requires_grad=False) for p in fac)
            self.factors.append(facModule)

        self.loaded_factors_to_laplace = False # true if self.laplace.H contains the same info as self.factors

    @property
    def sigma_noise(self):
        return torch.exp(self.log_sigma_noise)
    
    @property
    def prior_prec(self):
        return torch.exp(-self.log_prior_sig)

    @property
    def prior_sig(self):
        return torch.exp(self.log_prior_sig)
    
    @property
    def temp(self):
        return torch.exp(self.log_temp)

    def process_dataset(self, dataset):
        """
        summarizes information about training data by logging gradient directions
        seen during training, and then using gram schmidt of these to form
        an orthonormal basis. directions not seen during training are 
        taken to be irrelevant to data, and used for detecting generalization
        
        dataset - torch dataset of (input, target) pairs
        """
        self.laplace.H = None
        # loop through data as many times as we need to get 
        # num_samples of the weights with itr_between_samples
        print("computing basis")
        N = len(dataset)

        dataloader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size=self.batch_size, 
                                                 shuffle=True)

        self.laplace.fit(dataloader, keep_factors=True)
        self.loaded_factors_to_laplace = True

        self.store_factors(self.laplace.H_facs)
        self.configured.data = torch.ones(1, dtype=torch.bool)
    
    def store_factors(self, H_facs):
        """
        takes a list of Kron matrices, and stores all relevant data to nn.Parameters 
        so that they are saved with the model
        """
        for i, fac in enumerate(H_facs.kfacs):
            for j, val in enumerate(fac):
                self.factors[i][j].data = val

        self.n_data.data = torch.Tensor([self.laplace.n_data])
        self.n_outputs.data = torch.Tensor([self.laplace.n_outputs])
        self.loss.data = torch.Tensor([ self.laplace.loss] )

    def load_factors(self, damping=None):
        """
        Reconstructs self.laplace.H (which is not saved) from data in self.factors (which is saved)
        """
        if damping is None:
            damping = self.config['damping']

        H_facs_list = list()
        for i, weight in enumerate(self.factors):
            H_facs_list.append([fac for fac in weight])
        H_facs = Kron(H_facs_list)
        self.laplace.H = H_facs.decompose(damping=damping)

        self.laplace.n_data = self.n_data.data.item()
        self.laplace.n_outputs = self.n_outputs.data.item()
        self.laplace.loss = self.loss.data.item()

        self.laplace.prior_precision = self.prior_prec.cpu()
        self.loaded_factors_to_laplace = True

    def forward(self, inputs, verbose=False, damping=None):
        """
        assumes inputs are of shape (N, input_dims...)
        where N is the batch dimension,
              input_dims... are the dimensions of a single input
                            
        returns 
            mu = model(inputs) -- shape (N, d)
            unc = hessian based uncertainty estimates shape (N)
        """
        if not self.configured:
            print("Must call process_dataset first before using model for predictions.")
            raise NotImplementedError

        if not self.loaded_factors_to_laplace:
            self.load_factors(damping=damping)
        
        f_mu, f_var = self.laplace._glm_predictive_distribution(inputs)
        f_var = torch.diagonal(f_var, dim1=-2, dim2=-1)

        unc = []
        dists = []

        for j in range(f_mu.shape[0]):
            dist = self.dist_constructor(f_mu[j,...])
            output_dist = dist.marginalize(f_var[j,...])

            dists.append( output_dist )
            unc.append( (output_dist.entropy() ).sum())

        unc = torch.stack(unc)
    
        return dists, unc

    def optimize_nll(self, 
                     dataset : torch.utils.data.Dataset,
                     num_epochs : int = 1,
                     batch_size : int = 20):
        """
        tunes prior variance scale (eps) via SGD to minimize 
        validation nll on a given dataset
        """
        if not self.loaded_factors_to_laplace:
            self.load_factors()

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=4, pin_memory=True)
                                    
        self.laplace.optimize_prior_precision(method='CV', val_loader=dataloader, grid_size=10, verbose=True)

        self.log_prior_sig.data = -torch.log(self.laplace.prior_precision)