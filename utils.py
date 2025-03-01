import torch
import numpy as np
import os
from torch.utils.data import DataLoader
import argparse

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
# from backbone import ConvNet, Conv4, Conv4NP, ResNet18
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
    # print(scores)
    index = np.argsort(scores)
    select_feature = base_feature[index[0]]
    theta = scores[index[0]]
    # print(theta)
    theta = np.sqrt((1 + theta) / 2)
    interpolation = (1.0 / (2 * theta)) * feature + (1.0 / (2 * theta)) * select_feature
    # print(np.linalg.norm(feature))
    # print(np.linalg.norm(interpolation))
    return interpolation

def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    print("pretrain_dict:",pretrained_dict.keys())
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    print("model_dict:",model_dict.keys())
    model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
    model.load_state_dict(model_dict)

    return model


def feature_transform(support_feature, base_feature):
    for i in range(len(support_feature)):
        # feature = max_similarity(support_feature[i], base_feature, 'cos')
        # if feature:
        #     index, select_feature = feature
        #     # support_feature[i] = support_feature[i]
        #     support_feature[i] = 0.8 * support_feature[i]
        #     for j in range(len(select_feature)):
        #         support_feature[i] += 0.2 * select_feature[j]
        # else:
        #     support_feature[i] = support_feature[i]
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
            # support_copy /= np.linalg.norm(support_copy)
            # base_copy /= np.linalg.norm(base_copy)
            distance = -((support_copy - base_copy) ** 2).sum() / 64
            # print(distance)
            similarity[i] = distance

    k = 1
    # similarity = similarity[similarity > np.cos(45*np.pi/180)]
    # if len(similarity) == 0:
    #     return
    index = np.argsort(similarity)
    # print(index)
    print(similarity[index])
    index = index[len(base_feature) - k: len(base_feature)]
    return index, similarity[index], torch.from_numpy(base_feature[index]).cuda()

# def visualize(feature, label):
# model_dict = {
#     'Conv4': backbone.Conv4,
#     'Conv6': backbone.Conv6,
#     'ResNet10': backbone.ResNet10,
#     'ResNet18': backbone.ResNet18,
#     'ResNet34': backbone.ResNet34,
#     'ResNet12': resnet12.ResNet12}


if __name__ == '__main__':
    model_path = ''
    pretrained_dict = torch.load(model_path)
