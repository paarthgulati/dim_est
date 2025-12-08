import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


#for infinite data: fresh dataset of batch_size is generated every iteration, using the accompanying data_generator function

def train_model_infinite_data(model, data_generator, training_cfg: dict, optimizer_cls=torch.optim.Adam, device = 'cuda'):

    batch_size = training_cfg["batch_size"]
    n_iter = training_cfg["n_iter"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    show_progress = training_cfg["show_progress"]

    model.to(device)  
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi = []

    iterator = tqdm(range(n_iter)) if show_progress else range(n_iter)

    for i in iterator:
        x, y = data_generator(batch_size)
        
        opt.zero_grad()
        
        # Compute loss based on model type
        loss, mi, extras = model(x, y) 
        loss.backward()

        opt.step()

        estimator_tr = mi.to('cpu').detach().numpy()
        estimates_mi.append(estimator_tr)

    return np.array(estimates_mi)



def train_model_finite_data(model, train_data_loader, evalSet_X, evalSet_Y, testSet_X, testSet_Y,  training_cfg: dict, optimizer_cls=torch.optim.Adam, device = 'cuda'):

    batch_size = training_cfg["batch_size"]
    n_epoch = training_cfg["n_epoch"]
    lr = training_cfg["lr"]
    optimizer_kwargs = training_cfg.get("optimizer_kwargs", {})
    show_progress = training_cfg["show_progress"]

    model.to(device)  
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    estimates_mi_train = []
    estimates_mi_test = []

    iterator = tqdm(range(n_epoch)) if show_progress else range(n_epoch)

    for epochs in iterator:
        model.train()
        # train over all batches
        for i, (x,y) in enumerate(train_data_loader):
            opt.zero_grad()
            loss, mi, extras = model(x, y) 
            loss.backward()
            opt.step()

        # evaluate the mdoel at every epoch with a test and eval set (eval is a fixed subset of the training dataset):
        model.eval()
        with torch.no_grad():
            loss_train, mi_train, extras_train = model(evalSet_X, evalSet_Y) 
            loss_test, mi_test, extras_test = model(testSet_X, testSet_Y)

            mi_train = mi_train.to('cpu').detach().numpy()
            mi_test = mi_test.to('cpu').detach().numpy()

            estimates_mi_train.append(mi_train)
            estimates_mi_test.append(mi_test)

        
    return estimates_mi_train, estimates_mi_test
