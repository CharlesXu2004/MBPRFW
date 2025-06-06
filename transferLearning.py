from args import get_parser_transfer
import os.path as osp
import minisom
import numpy as np
import os
import torch
from transferUtils import *
import torch.nn.functional as F
import torch.nn as nn
from dataset import CIFAR, FC100, CUB, MiniImageNet
from samplers import CategoriesSampler
from backbones import backbone
from backbones import resnet12
from model import BaseFinetine
from tqdm import tqdm

model_dict = {
    'Conv4': backbone.Conv4,
    'Conv6': backbone.Conv6,
    'ResNet10': backbone.ResNet10,
    'ResNet18': backbone.ResNet18,
    'ResNet34': backbone.ResNet34,
    'ResNet12': resnet12.ResNet12
}

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    parser = get_parser_transfer()
    args = parser.parse_args()
    cum_episode = 0

    test_loader, base_class = data_load(args)
    base_model, model_name = basemodel_load(args, base_class, args.model_path)

    if(args.SOM):
        device_som = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net_SOM = resnet12.overall_model_som().to(device_som)
        optimizer_som = torch.optim.SGD(net_SOM.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
        lr_scheduler_som = torch.optim.lr_scheduler.StepLR(optimizer_som, step_size=args.step_size, gamma=args.gamma)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = resnet12.overall_model().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    accAll=[]
    accAll_som = []
    for i in range(7):
        accAll.append([])
        accAll_som.append([])

    for batch in tqdm(test_loader):

        initial_mat = torch.from_numpy(np.eye(640, dtype=np.float32)).cuda()
        net.train_w[0].weight.data = initial_mat
        if(args.SOM):
            net_SOM.train_w[0].weight.data = initial_mat

        data, label = [_.cuda() for _ in batch]
        data = base_model(data)
        if (args.BN == True):
            data = F.normalize(data, dim=1)
        
        p = args.shot * args.test_way
        data_shot, data_query = data[:p], data[p:]
        img1_1 = data_shot[0].unsqueeze(0)
        if (args.shot == 1):
            img1_2 = data_shot[0].unsqueeze(0)
        else:
            img1_2 = data_shot[5].unsqueeze(0)
        img0_1 = data_shot[0].unsqueeze(0)
        img0_2 = data_shot[1].unsqueeze(0)

        for i in range(p):
            for j in range(i + 1, p):
                if (label[i] == label[j]):
                    img1_1 = torch.cat((img1_1, data_shot[i].unsqueeze(0)), 0)
                    img1_2 = torch.cat((img1_2, data_shot[j].unsqueeze(0)), 0)
                else:
                    img0_1 = torch.cat((img0_1, data_shot[i].unsqueeze(0)), 0)
                    img0_2 = torch.cat((img0_2, data_shot[j].unsqueeze(0)), 0)

        if(args.SOM):
            img1_1_som = img1_1.clone().detach()
            img1_2_som = img1_2.clone().detach()
            img0_1_som = img0_1.clone().detach()
            img0_2_som = img0_2.clone().detach()

        for epoch in range(0, args.max_epoch + 1):

            if epoch==0:
                pass
            else:
                net.train()
                running_corrects = 0.0

                feature1_1, feature1_2 = net(img1_1, img1_2)
                feature0_1, feature0_2 = net(img0_1, img0_2)
                output0 = torch.cosine_similarity(feature0_1, feature0_2)
                output1 = torch.cosine_similarity(feature1_1, feature1_2)
                output0 = ((output0+1)/2).view(-1,1)
                output1 = ((output1+1)/2).view(-1,1)
                label0 = torch.zeros_like(output0)
                label1 = torch.ones_like(output1)
                preds0 = output0>0.5
                preds1 = output1>0.5
                loss0 = nn.BCELoss()(output0, label0)
                if(args.shot==1):
                    loss1 = 0
                else:
                    loss1 = nn.BCELoss()(output1, label1)
                loss = loss1 + loss0
                running_corrects = (preds0 == label0).type(torch.cuda.FloatTensor).mean().item()
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                # print('epoch {}, , loss={:.4f}'
                #       .format(epoch, loss.item()))
                # print('running_corrects:', running_corrects)
                net.eval()

                if(args.SOM):
                    net_SOM.train()
                    feature1_1_som, feature1_2_som = net_SOM(img1_1_som, img1_2_som)
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
                        t = abs(som.winner(img0_1[i].cpu().detach().numpy())[0]-som.winner(img0_2[i].cpu().detach().numpy())[0])
                        loss0_som += nn.BCELoss()(output0_som[i].unsqueeze(0), label0[i].unsqueeze(0))*(5/(t+1))

                    if (args.shot == 1):
                        loss1_som = 0
                    else:
                        loss1_som = nn.BCELoss()(output1_som, label1)

                    loss_som = loss1_som + loss0_som
                    running_corrects_som = (preds0_som == label0).type(torch.cuda.FloatTensor).mean().item()
                    optimizer_som.zero_grad()
                    loss_som.backward(retain_graph=True)
                    optimizer_som.step()
                    # print('loss_som={:.4f}'
                    #       .format(loss_som.item()))
                    # print('running_corrects_som:', running_corrects_som)
                    net_SOM.eval()


            with torch.no_grad():
                data_shot,data_query = net(data_shot,data_query)

                label = torch.arange(args.test_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)
                acc = evaluate(data_query, data_shot, label, args.shot, args.BN)
                acc_som=0
                if(args.SOM):
                    data_shot_som, data_query_som = net_SOM(data_shot, data_query)
                    acc_som = evaluate(data_query_som, data_shot_som, label, args.shot, args.BN)

                for i in range(7):
                    if(epoch==i*10):
                        accAll[i].append(acc)
                        if (args.SOM):
                            accAll_som[i].append(acc_som)

                # print("acc_som={:.4f}----acc={:.4f}".format(acc_som,acc))
        
        cum_episode += 1
        # if cum_episode == 1:
        #     break

    print("shot={}\nlr={}\nBN={}\nepisode={}".format(args.shot, args.lr, args.BN, cum_episode))
    for i in range(7):
        print("acc"+str(i*10)+"={:.4f} +- {:.4f}".format(np.mean(np.asarray(accAll[i]))
                                                         , 1.96 * (np.std(np.asarray(accAll[0])) / np.sqrt(len(np.asarray(accAll[0]))))))
        if (args.SOM):
            print("acc"+str(i*10)+"_som={:.4f} +- {:.4f}".format(np.mean(np.asarray(accAll_som[i]))
                                                         , 1.96 * (np.std(np.asarray(accAll_som[i])) / np.sqrt(len(np.asarray(accAll_som[i]))))))






