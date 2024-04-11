#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from CKA import linear_CKA, kernel_CKA
from collections import OrderedDict
import torch.nn.functional as F


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)

def crossentropyloss_between_logits(y_pred_logit, y_true_logit):
    return torch.mean(
        -torch.sum(
            F.softmax(y_pred_logit, dim=1) * y_true_logit, dim=1
        )
    )

class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.idx = idxs
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round,idx,flash):

        model.train()
        epoch_loss = []


        if global_round>=1:
            if flash==3:
                for child in enumerate(model.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 or child[0]==5 or child[0]==6:
                        for param in child[1].parameters():
                            param.requires_grad = False
            elif flash==2:
                for child in enumerate(model.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 or child[0]==5 :
                        for param in child[1].parameters():
                            param.requires_grad = False
            elif flash==1:
                for child in enumerate(model.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 :
                        for param in child[1].parameters():
                            param.requires_grad = False


        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9,
                              weight_decay=5e-4)
        gradients = []
        accumulated_gradients = None 
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 50 == 0):
                    print('clients :{} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        idx,
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        new_state_dict = OrderedDict()
        return model.state_dict(), sum(epoch_loss) / len(
            epoch_loss), model, accumulated_gradients, model.state_dict()


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


class LocalUpdate_shrinking(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))
        self.idx = idxs
        self.device = 'cuda' if args.gpu else 'cpu'
        # Default criterion set to NLL loss function
        # self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = nn.MSELoss().to(self.device)
        # self.criterion = crossentropyloss_between_logits.to(self.device)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(1*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round,idx,capacity,flash,model_shrinking):

        model.train()
        model_shrinking.train()
        epoch_loss = []

        if global_round>=1:
            if flash==3:
                for child in enumerate(model_shrinking.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 or child[0]==5 or child[0]==6:
                        for param in child[1].parameters():
                            param.requires_grad = False
            elif flash==2:
                for child in enumerate(model_shrinking.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 or child[0]==5 or child[0]==7 or child[0]==8 or child[0]==9:
                        for param in child[1].parameters():
                            param.requires_grad = False
            elif flash==1:
                for child in enumerate(model_shrinking.children()):
                    if child[0]==0 or child[0]==1 or child[0]==2 or child[0]==3 or child[0]==4 or child[0]==6 or child[0]==7 or child[0]==8 or child[0]==9:
                        for param in child[1].parameters():
                            param.requires_grad = False

        optimizer = torch.optim.SGD(model_shrinking.parameters(), lr=0.00001, momentum=0.9,
                              weight_decay=5e-4)

        accumulated_gradients = None 
        for iter in range(5):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):

                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                model_shrinking.zero_grad()
                log_probs = model(images)
                prob_shrinking = model_shrinking(images)
                loss_shrinking = self.criterion(prob_shrinking,log_probs)


                # loss = self.criterion(log_probs, labels)
                loss_shrinking.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 50 == 0):
                    print('------------------------global model shrinking---------------------------')
                    print('clients :{} | Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        idx,
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss_shrinking.item()))
                self.logger.add_scalar('loss', loss_shrinking.item())
                batch_loss.append(loss_shrinking.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        new_state_dict = OrderedDict()
        return model_shrinking.state_dict(), sum(epoch_loss) / len(
            epoch_loss), model_shrinking, accumulated_gradients, model_shrinking.state_dict()


    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def get_activations(model, inputs,activations):
    x = inputs

    for name,module in model.named_children():

        if name == 'fc':
            x = x.view(-1,512)
        x = module(x)

        if name in activations and activations[name] is None:
            activations[name] = x
        elif name in activations and activations[name] is not None:
            activations[name] = torch.cat([activations[name],x],dim=0)
    return activations

def test_inference_init(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss

def test_inference(args, model,reference_model ,test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    reference_model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    all_CKA = []

    activations_model = 0
    activations_reference = 0  

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=64,
                            shuffle=False)
    model_activations = {'layer1': None, 'layer2': None, 'layer3': None, 'layer4': None}
    ref_activations = {'layer1': None, 'layer2': None, 'layer3': None, 'layer4': None}

    flag = 0
    for batch_idx, (images, labels) in enumerate(testloader):
        # if batch_idx>=2:
        #     break
        images, labels = images.to(device), labels.to(device)
        flag = flag+1
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        if batch_idx%5==0:
     
            activations_model= (get_activations(model, images, model_activations))

 
            activations_reference = (get_activations(reference_model, images, ref_activations))
        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    loss = loss/flag
    #compute conv1 activation
    model_conv1_act = activations_model['layer1']
    avg_model_conv1_act = torch.mean(model_conv1_act,dim=(2,3))
    model_conv2_act = activations_model['layer2']
    avg_model_conv2_act = torch.mean(model_conv2_act, dim=(2, 3))
    model_conv3_act = activations_model['layer3']
    avg_model_conv3_act = torch.mean(model_conv3_act, dim=(2, 3))
    model_conv4_act = activations_model['layer4']
    avg_model_conv4_act = torch.mean(model_conv4_act, dim=(2, 3))

    ref_conv1_act = activations_reference['layer1']
    refavg_conv1_act = torch.mean(ref_conv1_act,dim=(2,3))
    ref_conv2_act = activations_reference['layer2']
    refavg_conv2_act = torch.mean(ref_conv2_act, dim=(2, 3))
    ref_conv3_act = activations_reference['layer3']
    refavg_conv3_act = torch.mean(ref_conv3_act, dim=(2, 3))
    ref_conv4_act = activations_reference['layer4']
    refavg_conv4_act = torch.mean(ref_conv4_act, dim=(2, 3))

    conv1_CKA = linear_CKA(avg_model_conv1_act.detach().cpu().numpy(), refavg_conv1_act.detach().cpu().numpy())
    conv2_CKA = linear_CKA(avg_model_conv2_act.detach().cpu().numpy(), refavg_conv2_act.detach().cpu().numpy())
    conv3_CKA = linear_CKA(avg_model_conv3_act.detach().cpu().numpy(), refavg_conv3_act.detach().cpu().numpy())
    conv4_CKA = linear_CKA(avg_model_conv4_act.detach().cpu().numpy(), refavg_conv4_act.detach().cpu().numpy())

    accuracy = correct/total
    return accuracy, loss, [conv1_CKA,conv2_CKA,conv3_CKA,conv4_CKA]
