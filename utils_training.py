import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


#for infinite data: fresh dataset of batch_size is generated every iteration, using the accompanying data_generator function

def train_model_infinite_data(model, data_generator, batch_size, n_iter, device = 'cuda', show_progress = False, optimizer_cls=torch.optim.Adam, lr=5e-4, optimizer_kwargs=None):

    model.to(device)  
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi = []

    iterator = tqdm(range(n_iter)) if show_progress else range(n_iter)

    for i in iterator:
        zx, zy = data_generator(batch_size)
        
        opt.zero_grad()
        
        # Compute loss based on model type
        mi, mi_i1, mi_i2 = model(zx, zy) 
        mi.backward()

        opt.step()

        estimator_tr = mi.to('cpu').detach().numpy()
        estimates_mi.append(estimator_tr)

    return np.array(estimates_mi)
