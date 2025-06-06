import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from args import get_parser_test
import torch.nn.functional as F
import minisom
import numpy as np
import torch
from torch.utils.data import DataLoader
from dataset import CIFAR, FC100, CUB, MiniImageNet
from utils import load_model
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12
from BaseModel import BaseModel
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12
}

if __name__ == '__main__':
    parser = get_parser_test()
    args = parser.parse_args()

    if 'Conv' in args.backbone or 'ResNet12' in args.backbone:
        image_size = 84
    else:
        image_size = 224

    if args.dataset == 'mini':
        testset = MiniImageNet('test', image_size, False)
        trainset = MiniImageNet('train', image_size, False)
        base_class = 64
    elif args.dataset == 'CUB':
        testset = CUB('test', image_size, False)
        trainset = CUB('train', image_size, False)
        base_class = 100
    elif args.dataset == 'FC100':
        testset = FC100('test', image_size, False)
        trainset = FC100('train', image_size, False)
        base_class = 60
    elif args.dataset == 'CIFAR':
        testset = CIFAR('test', image_size, False)
        trainset = CIFAR('train', image_size, False)
        base_class = 64

    test_sampler = CategoriesSampler(testset.label, 2000, args.test_way, args.shot + args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler, num_workers=1, pin_memory=True)
    train_loader = DataLoader(dataset=trainset, batch_size=256, shuffle=True, num_workers=1, pin_memory=True)

    backbone = model_dict[args.backbone]()
    model = BaseModel(backbone, base_class, args.test_way)

    model_path = './save/model_best.pth'
    model_path = '/data/user7/XuWei/MBPRFW/MBPRFW/idea3-1/save/basefinetineResNet12mini_0.8/maxacc.pth'
    if os.path.exists(model_path):
        load_model(model, model_path)
        print("load pretrain model successfully")

    model.cuda()
    model.eval()

    # # train SOM with base set
    # som = minisom.MiniSom(8, 8, 640)
    # for i, batch in enumerate(tqdm(train_loader, desc="Training SOM")):
    #     data, train_label = [_.cuda() for _ in batch]
    #     feature = F.normalize(model(data), dim=1)
    #     feature = feature.cpu().detach().numpy()
    #     som.train(feature, 2000)

    accuracy = []
    acc = 0
    accuracy_base = []
    acc_base = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Testing")):
            data, labelori = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            label = torch.arange(args.test_way).repeat(args.query).cuda()

            # acc = model.evaluate(data_query,data_shot,label,args.shot)
            # acc = model.cluster_evaluate(data,data_query, data_shot, label, args.shot)
            acc = model.cluster5_evaluate(data, data_query, data_shot, label, args.shot)
            # acc, acc_base = model.cluster5_evaluate_end(data, data_query, data_shot, label, args.shot)
            # acc, acc_base = model.cluster5_evaluate_protype(data, data_query, data_shot, label, args.shot)

            accuracy.append(acc)
            accuracy_base.append(acc_base)

    accuracy = np.asarray(accuracy)
    print("acc={:.4f} +- {:.4f}".format(np.mean(accuracy), 1.96 * (np.std(accuracy) / np.sqrt(len(accuracy)))))

    accuracy_base = np.asarray(accuracy_base)
    print("acc_base={:.4f} +- {:.4f}".format(np.mean(accuracy_base), 1.96 * (np.std(accuracy_base) / np.sqrt(len(accuracy_base)))))