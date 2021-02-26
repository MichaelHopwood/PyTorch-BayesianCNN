from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

#import data
from data import graph_data
import utils
import metrics
import config_graph_frequentist as cfg
from models.NonBayesianModels.SGC import SGConv

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
                   allow_zero_in_degree=False):

    if (net_type == 'sgc'):
        net =  SGConv(in_feats,
                        n_classes,
                        k=k,
                        cached=cached,
                        bias=bias,
                        norm=norm,
                        allow_zero_in_degree=allow_zero_in_degree)
        print("net",net)
        return net

def run(dataset, net_type):

    print(dataset, net_type)
    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    g, features, labels, in_feats, n_classes, n_edges, train_mask, val_mask, test_mask = graph_data.getDataset(dataset)

    net = getModel(net_type,
                    in_feats,
                    n_classes,
                    k=1,
                    cached=False,
                    norm=None,
                    bias=True,
                    allow_zero_in_degree=False)
    net = net.to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs+1):

        # TRAIN
        net.train()
        print("features",features.numpy().shape)
        logits = net(g, features) # only compute the train set
        labels = labels.type(torch.LongTensor)
        #print('logits:',logits.detach().numpy().shape)
        #print('labels:',labels.numpy().shape)
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


        # VALIDATE
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

    # Test
    net.eval()
    # VALIDATE ACCURACY AND LOSS
    with torch.no_grad():
        logits = net(g, features)[test_mask] # only compute the evaluation set
        loss = criterion(logits, labels[test_mask])
        _, indices = torch.max(logits, dim=1)
        try:
            correct = torch.sum(indices == labels)
        except Exception as e:
            correct = torch.sum(torch.LongTensor(indices) == labels[test_mask])
        test_acc = correct.item() * 1.0 / len(labels[test_mask])
    test_loss = loss.item()

    print('\n\nTest Loss: {:.4f} \tTest Accuracy: {:.4f}'.format(
            test_loss, test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='sgc', type=str, help='model')
    parser.add_argument('--dataset', default='cora', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
