import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import argparse

from dataset import MiniImageNet
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


def cos(feature, base_feature):
    scores = []
    for i in range(len(base_feature)):
        fea = base_feature[i]
        score = np.dot(feature, fea)
        scores.append(score)
    index = np.argsort(scores)
    select_feature = base_feature[index[0]]
    theta = scores[index[0]]
    theta = np.sqrt((1 + theta) / 2)
    interpolation = (1.0 / (2 * theta)) * feature + (1.0 / (2 * theta)) * select_feature
    return interpolation


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

    return model


def feature_transform(support_feature, base_feature):
    for i in range(len(support_feature)):
        index, similarity, select_feature = max_similarity(support_feature[i], base_feature, 'euclidean')
        support_feature[i] = 0.5 * support_feature[i]
        for j in range(len(select_feature)):
            support_feature[i] += 0.5 * select_feature[j]

    return support_feature


def max_similarity(support_feature, base_feature, measure):
    similarity = np.zeros([len(base_feature), ])
    for i in range(len(base_feature)):
        support_copy = support_feature.detach().cpu().numpy()
        base_copy = base_feature[i]
        if measure == 'cos':
            a_norm = np.linalg.norm(support_copy)
            b_norm = np.linalg.norm(base_copy)
            cos = np.dot(support_copy, base_copy) / (a_norm * b_norm)
            similarity[i] = cos

        if measure == 'euclidean':
            distance = -((support_copy - base_copy) ** 2).sum() / 64
            similarity[i] = distance

    k = 1
    index = np.argsort(similarity)
    print(similarity[index])
    index = index[len(base_feature) - k: len(base_feature)]
    return index, similarity[index], torch.from_numpy(base_feature[index]).cuda()
