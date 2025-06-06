import argparse
import os

import torch
from torch.utils.data import DataLoader
from backbones import resnet12
from model import BaseFinetine
from dataset import CIFAR, FC100, CUB, MiniImageNet
from samplers import CategoriesSampler
from utils import load_model
import torch.nn.functional as F


def data_load(args):
    image_size = 84
    if args.dataset == 'mini':
        testset = MiniImageNet('test', image_size, False)
        base_class = 64
    elif args.dataset == 'CUB':
        testset = CUB('test', image_size, False)
        base_class = 100
    elif args.dataset == 'FC100':
        testset = FC100('test', image_size, False)
        base_class = 60
    elif args.dataset == 'CIFAR':
        testset = CIFAR('test', image_size, False)
        base_class = 64

    test_sampler = CategoriesSampler(testset.label, 1000, args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=1, pin_memory=True)
    return test_loader,base_class


def basemodel_load(args, base_class, model_path):
    backbone = resnet12.ResNet12()
    base_model = BaseFinetine(backbone, base_class, args.test_way)
    if os.path.exists(model_path):
        load_model(base_model, model_path)
        print("load pretrain model successfully")
    base_model.cuda()
    base_model.eval()
    return base_model, model_path


def evaluate(data_query, data_shot, label, n, BN):
    if n == 5:
        data_shot = data_shot.reshape(5, 5, 640)
        data_shot = torch.mean(data_shot, dim=0)
    if(BN==False):
        data_query = F.normalize(data_query, dim=1)
        data_shot = F.normalize(data_shot, dim=1)
    score = torch.matmul(data_query, data_shot.T)
    pred = torch.argmax(score, dim=1)

    acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    return acc
