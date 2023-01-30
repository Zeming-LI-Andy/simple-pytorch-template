import os

import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim

import Utils

from preprocess.Dataset import get_dataloader
from model.Models import MyModel
from tqdm import tqdm


def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

    def load_data(path_name):
        data = []
        with open(path_name, 'r') as fp:
            for line in fp:
                data.append(line.strip().split('\t'))
        return data

    print('[Info] Loading data...')
    train_data = load_data()
    test_data = load_data()

    trainloader = get_dataloader(train_data, opt.batch_size, shuffle=True)
    testloader = get_dataloader(test_data, opt.batch_size, shuffle=False)
    return trainloader, testloader


def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()
    total_loss = 0
    size = len(training_data)
    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, event_type = map(lambda x: x.to(opt.device), batch)

        """ forward """
        optimizer.zero_grad()

        prediction = model(event_type, event_time)

        """ backward """
        # negative log-likelihood
        loss = ()
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """
        total_loss += loss.detach().item()

    return total_loss/size


def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()

    total_loss = 0
    size = len(validation_data)
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """
            event_time, event_type = map(lambda x: x.to(opt.device), batch)

            """ forward """
            prediction = model(event_type, event_time)

            """ compute loss """
            loss = ()

            """ note keeping """
            total_loss += loss.detach().item()
            
    return total_loss/size


def train(model, training_data, validation_data, optimizer, scheduler, pred_loss_func, opt):
    """ Start training. """

    valid_avg_losses = []  # validation log-likelihood
    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()
        train_avg_loss = train_epoch(model, training_data, optimizer, pred_loss_func, opt)
        print('  - (Training)    average_loss: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_avg_loss, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_avg_loss = eval_epoch(model, validation_data, pred_loss_func, opt)
        print('  - (Testing)     loglikelihood: {ll: 8.5f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=valid_avg_loss, elapse=(time.time() - start) / 60))
        valid_avg_losses.append(valid_avg_loss)

        print('  - [Info] Minimum ll: {event: 8.5f}'
              .format(event=min(valid_avg_losses)))

        # logging
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {ll: 8.5f}\n'
                    .format(epoch=epoch, ll=valid_avg_loss))
        
        # model params
        params = {
            'state_dict': model.state_dict(), 
            'dropout': model.dropout}
        torch.save(params, opt.params + f'model_epoch_{epoch}.pth')

        scheduler.step()


def main():
    """ Main function. """

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', type=str, default='/home/comp/cszmli/models/data/hawkes/')

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-batch_size', type=int, default=32)

    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-smooth', type=float, default=0.1)

    parser.add_argument('-log', type=str, default='esa_log.txt')
    parser.add_argument('-params', type=str, default='./esa_params/')
    
    parser.add_argument('-pre_trained_model', type=str, default='./eur_params/transformer_hawkes_epoch_36.pth')    

    opt = parser.parse_args()

    # default device is CUDA
    opt.device = torch.device('cuda')

    # setup the log file
    with open(opt.log, 'w') as f:
        f.write('Epoch, Log-likelihood, Accuracy, RMSE\n')

    # setup params folder
    if os.path.exists(opt.params) is False:
        os.makedirs(opt.params)

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    trainloader, testloader, num_types = prepare_dataloader(opt)

    """ prepare model """
    if opt.pre_trained_model != '':
        params = torch.load(opt.pre_trained_model)
        model = MyModel()
        model.load_state_dict(params['state_dict'])
        print('[Info] Successfully loading model')
    else:
        model = MyModel()
    model.to(opt.device)

    """ optimizer and scheduler """
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.5)

    """ prediction loss function, either cross entropy or label smoothing """
    if opt.smooth > 0:
        pred_loss_func = Utils.LabelSmoothingLoss(opt.smooth, num_types, ignore_index=-1)
    else:
        pred_loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')

    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    train(model, trainloader, testloader, optimizer, scheduler, pred_loss_func, opt)


if __name__ == '__main__':
    main()
