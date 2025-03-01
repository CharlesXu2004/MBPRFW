# import argparse
# import os.path as osp
#
# import math
import minisom
# from torchvision.models import ResNet
import numpy as np
# import os
# import torch
from transferUtils import *
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
# from utils import load_model
# from CIFAR import CIFAR
# from FC100 import FC100
# from cub import CUB
# from mini_imagenet import MiniImageNet
# from samplers import CategoriesSampler
# # from backbone import ConvNet, Conv4, Conv4NP, ResNet18
# from backbones import backbone
from backbones import resnet12
# from basefinetine import BaseFinetine
# from torchvision import transforms
# from utils import TwoCropTransform


if __name__ == '__main__':
    args = arg_parameter()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    BN = False
    episode = 0

    test_loader, base_class = data_load(args)
    base_model,model_name = basemodel_load(args, base_class)


    device_som = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net_SOM = resnet12.overall_model_som().to(device_som)
    lr_som = args.lr
    optimizer_som = torch.optim.SGD(net_SOM.parameters(), lr=lr_som, momentum=0.9, nesterov=True,
                                    weight_decay=0.0005)
    lr_scheduler_som = torch.optim.lr_scheduler.StepLR(optimizer_som, step_size=args.step_size, gamma=args.gamma)



    accAll_som = []
    for i in range(7):
        accAll_som.append([])
    eps = 1e-5

    for j, batch in enumerate(test_loader):
        episode = j
        print('episode:',j,'-------------------------------------------------')
        a = np.eye(640, dtype=float)
        a = torch.from_numpy(np.float32(a)).cuda()
        net_SOM.train_w[0].weight.data = a

        for epoch in range(0, args.max_epoch + 1):
            data, label = [_.cuda() for _ in batch]
            data = base_model(data)
            if(BN==True):
                data = F.normalize(data, dim=1)
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]
            running_corrects = 0.0
            if epoch==0:
                pass
            else:
                net_SOM.train()
                img1_1 = data_shot[0].unsqueeze(0)
                if(args.shot==1):
                    img1_2 = data_shot[0].unsqueeze(0)
                else:
                    img1_2 = data_shot[5].unsqueeze(0)
                img0_1 = data_shot[0].unsqueeze(0)
                img0_2 = data_shot[1].unsqueeze(0)
                for i in range(p):
                    for j in range(i+1,p):
                        if(label[i]==label[j]):
                            img1_1 = torch.cat((img1_1,data_shot[i].unsqueeze(0)),0)
                            img1_2 = torch.cat((img1_2, data_shot[j].unsqueeze(0)),0)
                        else:
                            img0_1 = torch.cat((img0_1, data_shot[i].unsqueeze(0)),0)
                            img0_2 = torch.cat((img0_2, data_shot[j].unsqueeze(0)),0)




                feature1_1_som, feature1_2_som = net_SOM(img1_1.data, img1_2.data)
                feature0_1_som, feature0_2_som = net_SOM(img0_1.data, img0_2.data)
                output0_som = torch.cosine_similarity(feature0_1_som, feature0_2_som)
                output1_som = torch.cosine_similarity(feature1_1_som, feature1_2_som)
                output0_som = ((output0_som + 1) / 2).view(-1, 1)
                output1_som = ((output1_som + 1) / 2).view(-1, 1)
                label0 = torch.zeros_like(output0_som)
                label1 = torch.ones_like(output1_som)
                preds0_som = output0_som > 0.5
                preds1_som = output1_som > 0.5
                som = minisom.MiniSom(3, 4, 640)
                data_numpy = data.cpu().detach().numpy()
                som.train(data_numpy, 2000)
                loss0_som = 0
                for i in range(len(feature0_2_som)):
                    t = abs(som.winner(img0_1[i].detach().cpu().numpy())[0]-som.winner(img0_2[i].cpu().detach().numpy())[0])
                    loss0_som += nn.BCELoss()(output0_som[i].unsqueeze(0), (label0[i]).unsqueeze(0))*(5/(t+2))
                    #loss0_som += nn.BCELoss()(output0_som[i].unsqueeze(0), (label0[i]).unsqueeze(0)) * (output0_som[i] / 0.5)*1.5
                loss0_som /= len(feature0_2_som)
                loss1_som = 0
                for i in range(len(feature1_2_som)):
                    t = abs(
                        som.winner(img1_1[i].detach().cpu().numpy())[0] - som.winner(img1_2[i].cpu().detach().numpy())[
                            0])
                    loss1_som += nn.BCELoss()(output1_som[i].unsqueeze(0), (label1[i]).unsqueeze(0)) * ((t+2)/ 3)
                    # loss0_som += nn.BCELoss()(output0_som[i].unsqueeze(0), (label0[i]).unsqueeze(0)) * (output0_som[i] / 0.5)*1.5
                loss1_som /= len(feature1_2_som)

                # if (args.shot == 1):
                #     loss1_som = 0
                # else:
                #     loss1_som = nn.BCELoss()(output1_som, label1)

                loss_som = loss1_som + loss0_som
                running_corrects_som = (preds0_som == label0).type(torch.cuda.FloatTensor).mean().item()
                optimizer_som.zero_grad()
                loss_som.backward()
                optimizer_som.step()
                print('epoch {}, , loss_som={:.4f}'
                      .format(epoch, loss_som.item()))
                print('running_corrects_som:', running_corrects_som)
            net_SOM.eval()
            with torch.no_grad():
                data_shot_som, data_query_som = net_SOM(data_shot, data_query)
                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                acc_som = evaluate(data_query_som, data_shot_som, label, args.shot, BN)
                for i in range(7):
                    if(epoch==i*10):
                        accAll_som[i].append(acc_som)

                print("acc_som={:.4f}".format(acc_som))



    val = np.asarray(accAll_som[0])
    print("np.std(val):", np.std(val))
    print("model_name={}".format(model_name))
    print("shot={}______lr={}______训练集BN={}_____episode={}".format(args.shot,args.lr,BN,episode+1))
    for i in range(7):
        print("acc"+str(i*10)+"_som={:.4f} +- {:.4f}".format(np.mean(np.asarray(accAll_som[i]))
                                                         , 1.96 * (np.std(np.asarray(accAll_som[i])) / np.sqrt(len(val)))))






