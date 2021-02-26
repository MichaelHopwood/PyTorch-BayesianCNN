from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

#import data
from data import graph_data
import utils
import metrics
import config_graph_bayesian as cfg
from models.BayesianModels.BayesianSGC import BBBSGC

# CUDA settings
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

def getModel(net_type,
                   in_feats,
                   n_classes,
                   k=1,
                   cached=False,
                   norm=None,
                   bias=True,
                   allow_zero_in_degree=False,
                   layer_type='lrt'):

    if (net_type == 'sgc'):
        net =  BBBSGC(in_feats,
                        n_classes,
                        k=k,
                        cached=cached,
                        bias=bias,
                        norm=norm,
                        allow_zero_in_degree=allow_zero_in_degree,
                        layer_type=layer_type)
        print("net",net)
        return net

def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    test_ens = cfg.test_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    g, features, labels, in_feats, n_classes, n_edges, train_mask, val_mask, test_mask = graph_data.getDataset(dataset)
    
    n_train_samples = len(labels[train_mask])
    n_test_samples = len(labels[test_mask])
    n_valid_samples = len(labels[val_mask])

    print("n_train_samples",n_train_samples)
    print("n_test_samples",n_test_samples)

    net = getModel(net_type,
                    in_feats,
                    n_classes,
                    k=1,
                    cached=False,
                    norm=None,
                    bias=True,
                    allow_zero_in_degree=False,
                    layer_type=layer_type)
    net = net.to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(n_train_samples).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_min = np.Inf

    features_shape = features.numpy().shape
    for epoch in range(1, n_epochs+1):

        # TRAIN
        net.train()
        optimizer.zero_grad()

        outputs = torch.zeros(features_shape[0], n_classes, cfg.train_ens).to(device)

        kl = 0.0
        for j in range(cfg.train_ens):
            net_out, _kl = net(g, features)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / cfg.train_ens
        log_outputs = utils.logmeanexp(outputs, dim=2)

        # print('kl',kl)
        # print('log_outputs',log_outputs)
        # print('log_outputs shape',log_outputs.detach().numpy().shape)

        #zipped = zip(log_outputs[train_mask], labels[train_mask])
        #for i,(log_out, lbl) in enumerate(zipped):
        # print(log_out)
        beta = 0.1#metrics.get_beta(i-1, n_train_samples, beta_type, epoch, n_epochs)
        #print(beta)
        loss = criterion(log_outputs[train_mask], labels[train_mask], kl, beta)
        loss.backward()
        optimizer.step()

        train_acc = metrics.acc(log_outputs.data, labels)
        train_loss = loss.item()

        '''
        net.train()
        logits = net(g, features) # only compute the train set
        labels = labels.type(torch.LongTensor)
        loss = criterion(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, indices = torch.max(logits, dim=1)
        try:
            correct = torch.sum(indices == labels)
        except Exception as e:
            correct = torch.sum(torch.LongTensor(indices) == labels)
        train_acc = correct.item() * 1.0 / len(labels)
        train_loss = loss.item()
        '''


        # VALIDATE
        net.eval() # could be net.train(), as we have no layers which are effected

        outputs = torch.zeros(features_shape[0], n_classes, cfg.train_ens).to(device)

        kl = 0.0
        for j in range(cfg.train_ens):
            net_out, _kl = net(g, features)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)

        kl = kl / cfg.train_ens
        log_outputs = utils.logmeanexp(outputs, dim=2)

        # print('kl',kl)
        # print('log_outputs',log_outputs)
        # print('log_outputs shape',log_outputs.detach().numpy().shape)

        #zipped = zip(log_outputs[train_mask], labels[train_mask])
        #for i,(log_out, lbl) in enumerate(zipped):
        # print(log_out)
        beta = 0.1#metrics.get_beta(i-1, n_train_samples, beta_type, epoch, n_epochs)
        #print(beta)
        loss = criterion(log_outputs[val_mask], labels[val_mask], kl, beta)

        valid_acc = metrics.acc(log_outputs.data, labels)
        valid_loss = loss.item()

        '''
        net.eval()
        # VALIDATE ACCURACY AND LOSS
        with torch.no_grad():
            logits = net(g, features)[val_mask] # only compute the evaluation set
            loss = criterion(logits, labels[val_mask])
            _, indices = torch.max(logits, dim=1)
            try:
                correct = torch.sum(indices == labels)
            except Exception as e:
                correct = torch.sum(torch.LongTensor(indices) == labels[val_mask])
            valid_acc = correct.item() * 1.0 / len(labels[val_mask])
        valid_loss = loss.item()
        lr_sched.step(valid_loss)
        '''

        # train_loss = train_loss/len(train_loader.dataset)
        # valid_loss = valid_loss/len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(net.state_dict(), ckpt_name)
            valid_loss_min = valid_loss

    # Test on trained model
    net.eval() # could be net.train(), as we have no layers which are effected

    outputs = torch.zeros(features_shape[0], n_classes, cfg.test_ens).to(device)

    kl = 0.0
    for j in range(cfg.train_ens):
        net_out, _kl = net(g, features)
        kl += _kl
        outputs[:, :, j] = F.log_softmax(net_out, dim=1)

    kl = kl / cfg.test_ens
    log_outputs = utils.logmeanexp(outputs, dim=2)

    # print('kl',kl)
    # print('log_outputs',log_outputs)
    # print('log_outputs shape',log_outputs.detach().numpy().shape)

    #zipped = zip(log_outputs[train_mask], labels[train_mask])
    #for i,(log_out, lbl) in enumerate(zipped):
    # print(log_out)
    beta = 0.1#metrics.get_beta(i-1, n_train_samples, beta_type, epoch, n_epochs)
    #print(beta)
    loss = criterion(log_outputs[test_mask], labels[test_mask], kl, beta)

    test_acc = metrics.acc(log_outputs.data, labels)
    test_loss = loss.item()

    print('\n\nTest Loss: {:.4f} \tTest Accuracy: {:.4f}'.format(
            test_loss, test_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='sgc', type=str, help='model')
    parser.add_argument('--dataset', default='cora', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
