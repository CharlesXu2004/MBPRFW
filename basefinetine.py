import numpy.linalg.linalg
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
#from MyLinear import MyLinear
# from losses import simple_contrstive_loss
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from utils import cos
# import minisom
class BaseFinetine(nn.Module):
    def __init__(self, model_func, train_way, test_way):
        super().__init__()
        self.model = model_func
        self.train_way = train_way
        self.test_way = test_way
        self.final_dim = self.model.final_feat_dim
        #self.train_fc = MyLinear(self.final_dim, train_way)
        self.train_fc = nn.Linear(self.final_dim, train_way)
        self.test_fc = nn.Linear(self.final_dim, test_way)

        #refine
        a = np.eye(640, dtype=float)
        a = torch.from_numpy(np.float32(a))
        self.train_w = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            nn.BatchNorm1d(self.final_dim,eps = 0.001,momentum=0.1,affine=True)
        )
        self.train_w[0].weight.data = a
        #self.eq = nn.Linear(self.final_dim, 8)
        self.base_temperature = 0.1
        self.batch_ = 0
        self.LR = LogisticRegression()
        #self.dropout = nn.Dropout(p=0.5)
        # self.inv_head_0 = nn.Sequential(
        #     nn.Linear(640, 640),
        #     nn.BatchNorm1d(640),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(640, 640),
        #     nn.BatchNorm1d(640),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(640, 64)
        # )
        # self.inv_head_1 = nn.Sequential(
        #     nn.Linear(640, 640),
        #     nn.BatchNorm1d(640),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(640, 640),
        #     nn.BatchNorm1d(640),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(640, 64)
        # )
        self.eq = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 8)
        )
        #self.eq = nn.Linear(self.final_dim, 8)
    def train_rotation_crop(self, data_rotation,label,label_aug):
        feature = self.model(data_rotation)

        logits = self.train_fc(feature)
        eq_logit = self.eq(feature)

        eq_loss = F.cross_entropy(eq_logit, label_aug)
        en_loss = F.cross_entropy(logits, label)
        loss = 0.8 * eq_loss + en_loss


        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc
    def train_base(self, data,label):
        feature = self.model(data)
        #print(self.train_w[0].weight.data)

        logits = self.train_fc(feature)
        loss = F.cross_entropy(logits, label)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc
    # attention
    # def train_weight(self, data,label,n):
    #     feature = self.model(data)
    #     #print(self.train_w[0].weight.data)
    #
    #     feature = self.train_w(feature)
    #     for i in range(n):
    #         for j in range(i+1,n):
    #             label = int(label[i]==label[j])
    #             query_feature = F.normalize(feature[i], dim=1)
    #             support_feature = F.normalize(feature[j], dim=1)
    #             score = torch.matmul(query_feature, support_feature.T)
    #             pred = torch.argmax(score, dim=1)
    #
    #
    #
    #
    #     loss = F.cross_entropy(logits, label)
    #
    #     pred = torch.argmax(logits, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return loss, acc
    def forward(self, x):
        return self.model(x)

    def train_LR(self,sup,sup_label,query,query_label):

        lr = LogisticRegression().fit(sup,sup_label)
        logits = lr.predict(query)
        pred = (logits==query_label)
        acc = sum(pred)/75
        return acc


    def train_svm(self,sup,sup_label,query,query_label,n):
        # if n==5:
        #     sup = sup.reshape(5, 5, 640)
        #     sup = np.mean(sup, axis=0)
        #sup_label = np.array([0, 1, 2, 3, 4])
        svm_sup = svm.SVC()
        svm_sup.fit(sup,sup_label)
        logits = svm_sup.predict(query)

        pred = (logits==query_label)
        acc = sum(pred)/75
        return acc

    def train_knn(self,sup,query,query_label,n):
        if n==5:
            sup = sup.reshape(5, 5, 640)
            sup = np.mean(sup, axis=0)
        sup_label = np.array([0, 1, 2, 3, 4])
        k =1
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(sup,sup_label)
        logits = clf.predict(query)

        pred = (logits==query_label)
        acc = sum(pred)/75
        return acc


    def train_loop_eq(self, x, train_label):
        #----------------rotation----------------
        outs = None
        flag = 1
        for x_0 in x:
            x_90 = x_0.transpose(1, 2).flip(1).unsqueeze(0)
            x_180 = x_0.flip(1).flip(2).unsqueeze(0)
            x_270 = x_0.flip(1).transpose(1, 2).unsqueeze(0)
            if flag == 1:
                outs = torch.cat((x_0.unsqueeze(0), x_90, x_180, x_270), 0)
                flag = 0
            else:
                outs = torch.cat((outs, x_0.unsqueeze(0), x_90, x_180, x_270), 0)

        #-----------------crop_size---------------------
        outs = outs.cpu()
        out = None
        flag_1 = 1
        transform_norm = transforms.Compose([
            # transforms.functional.resize_crop(np.random.randint(28), 0, 56, 84, (84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        for x0 in outs:
            temp = transforms.ToPILImage()(x0)
            x1 = transforms.functional.resized_crop(temp,np.random.randint(28), 0, 56, 84, (84, 84))
            x1 = transform_norm(x1)
            x2 = transforms.functional.resized_crop(temp, 0,np.random.randint(28), 56, 84, (84, 84))
            x2 = transform_norm(x2)
            x3 = transforms.functional.resized_crop(temp, np.random.randint(28),np.random.randint(28), 56, 56, (84, 84))
            x3 = transform_norm(x3)
            x0 = transform_norm(temp)
            if flag_1==1:
                out = torch.cat((x0.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)),0)
                flag_1 = 0
            else:
                out = torch.cat((out,x0.unsqueeze(0),x1.unsqueeze(0),x2.unsqueeze(0),x3.unsqueeze(0)),0)

        #---------------base-----------------
        for j in range(len(x)):
            x[j] = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])(x[j])


        #---------------label-----------------
        en_label = (train_label.repeat(16).reshape(16,-1).t()).reshape(-1)
        label_rotation_temp = [_%16 for _ in range(len(out))]


        # print("en_label:",len(en_label),en_label)
        # print("out:",out.shape)
        # print("rotation:",label_rotation_temp)
        #-------------data_shuffle-------------
        index = torch.randperm(len(out))
        temp = index[0]
        data = out[temp].unsqueeze(0)
        label = [en_label[temp]]
        label_eq = [label_rotation_temp[temp]]
        for i in index:
            data = torch.cat((data,out[i].unsqueeze(0)),0)
            label.append(en_label[i])
            label_eq.append(label_rotation_temp[i])
        data = data.cuda()
        label = torch.tensor(label).cuda()
        label_eq = torch.tensor(label_eq).cuda()

        #----------logit-----------
        feature_loop_eq = self.model(data)
        feature  = self.model(x)
        logit = self.train_fc(feature)
        logits = self.train_fc(feature_loop_eq)
        eq_logit = self.eq(feature_loop_eq)

        #--------------compute_loss--------------
          #-----base_loss-------
        loss_x = F.cross_entropy(logit,train_label)
          #------eq_loss--------
        eq_loss = F.cross_entropy(eq_logit, label_eq)
          #-----data-aug-loss-------
        en_loss = F.cross_entropy(logits, label)
         #----all-loss-----
        loss = 0.1 * eq_loss + 0.15 * en_loss + loss_x

        pred_eq = torch.argmax(eq_logit, dim=1)
        acc_eq = (pred_eq == label_eq).type(torch.cuda.FloatTensor).mean().item()
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        pred_x = torch.argmax(logit,dim=1)
        acc_x = (pred_x == train_label).type(torch.cuda.FloatTensor).mean().item()

        return loss,loss_x,en_loss,eq_loss,acc_x, acc, acc_eq


    # def train_loop_eq(self, data,label_Non,x, label,label_eq):
    #     feature = self.model(x)
    #     feature_Non = self.model(data)
    #     #feature_eq = self.model(x)
    #     #feature = F.normalize(feature, p=2, dim=1)
    #     logit = self.train_fc(feature_Non)
    #     logits = self.train_fc(feature)
    #     #feature_eq = F.normalize(feature_eq,p=2,dim=1)
    #     eq_logit = self.eq(feature)
    #     loss_Non = F.cross_entropy(logit,label_Non)
    #     eq_loss = F.cross_entropy(eq_logit,label_eq)
    #     en_loss = F.cross_entropy(logits, label)
    #     loss = 0.1*eq_loss+0.15*en_loss+loss_Non
    #     pred_eq = torch.argmax(eq_logit, dim=1)
    #     acc_eq = (pred_eq == label_eq).type(torch.cuda.FloatTensor).mean().item()
    #     pred = torch.argmax(logits, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     pred_Non = torch.argmax(logit, dim=1)
    #     acc_Non = (pred_Non == label_Non).type(torch.cuda.FloatTensor).mean().item()
    #     return loss_Non,acc_Non,loss, acc ,acc_eq,en_loss,eq_loss

    def train_loop(self, x, label):
        feature = self.model(x)
        # print(feature.shape)
        #feature = F.normalize(feature, p=2, dim=1)
        logits = self.train_fc(feature)
        loss = F.cross_entropy(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc
    # def train_loop_trans(self, x, label,data_trans,outs,label_outs):
    #     feature = self.model(x)
    #     feature_trans =self.model(data_trans)
    #     outs = self.model(outs)
    #     logits = self.train_fc(feature)
    #     inv_0 = self.inv_head_1(feature)
    #     inv = self.inv_head_0(feature_trans)
    #     outs = self.inv_head_1(outs)
    #     loss_invs = 0
    #     for i in range(len(feature)):
    #         loss_inv = simple_contrstive_loss(label[i],inv_0[i], inv[3*i:3*i+3],outs,label_outs)
    #         loss_invs = loss_invs + loss_inv
    #     loss_invs = loss_invs/len(feature)
    #     loss_ce = F.cross_entropy(logits, label)
    #     loss = loss_ce + 0.1*loss_invs
    #     pred = torch.argmax(logits, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return loss, acc


    def train_contrastive(self, x, label, temperature):
        x = torch.cat([x[0].cuda(), x[1].cuda()], dim=0)
        batch_size = x.shape[0]
        feature = self.model(x)
        feature = F.normalize(feature, dim=1)
        label = label.cuda()
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast
        # print(logits)
        mask = mask.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )
        # print(logits_mask)
        mask = mask * logits_mask
        # indexs = ((mask == 1)).nonzero(as_tuple=True)
        # for row, col in zip(*indexs):
        #     logits[row][col] = logits[row][col] + 2.0

        exp_logits = torch.exp(logits) * logits_mask
        # print(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        # loss
        loss = (-(temperature / self.base_temperature) * mean_log_prob_pos).mean()

        return loss

    # def evaluate(self, data_query, data_shot, label):
    #     query_feature = self.model(data_query).cpu().numpy()
    #     support_feature = self.model(data_shot).cpu().numpy()
    #     for j in range(len(support_feature)):
    #         support_norm = np.linalg.norm(support_feature[j])
    #         support_feature[j] = support_feature[j] / support_norm
    #         # print(support_norm)
    #
    #     for j in range(len(query_feature)):
    #         query_norm = np.linalg.norm(query_feature[j])
    #         query_feature[j] = query_feature[j] / query_norm
    #         # print(query_norm)
    #     acc = 0
    #     predicts = []
    #     for j in range(len(query_feature)):
    #         distances = []
    #         for k in range(len(support_feature)):
    #             distance = ((query_feature[j] - support_feature[k]) ** 2).sum()
    #             distances.append(distance)
    #         predict = np.argmin(distances)
    #         predicts.append(predict)
    #
    #     for j in range(len(label)):
    #         if (label[j] == predicts[j]):
    #             acc += 1
    #
    #     acc = acc / len(label)
    #
    #     return acc
    def cluster_evaluate(self, data ,data_query, data_shot, label,n):
        data_feature = self.model(data)
        data_som = data_feature.cpu().numpy()
        query_feature = self.model(data_query)
        query_som = query_feature.cpu().numpy()
        support_feature = self.model(data_shot)
        support_som = support_feature.cpu().numpy()
        som = minisom.MiniSom(2,2,640)
        som.train(data_som,2000)

        if n == 5:
            data_shot = support_feature.reshape(5, 5, 640)
            support_feature = torch.mean(data_shot, dim=0)
            support_som = support_feature.cpu().numpy()

        query_feature = F.normalize(query_feature, dim=1)
        support_feature = F.normalize(support_feature, dim=1)
        score = torch.matmul(query_feature, support_feature.T)
        for i in range(len(query_som)):
            t = som.winner(query_som[i])
            for j in range(5):
                if(t!=som.winner(support_som[j])):
                    score[i][j]=0
        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc
    # def cluster5_evaluate(self, data ,data_query, data_shot, label,n):
    #     data_feature = self.model(data)
    #     data_som = data_feature.cpu().numpy()
    #     query_feature = self.model(data_query)
    #     query_som = query_feature.cpu().numpy()
    #     support_feature = self.model(data_shot)
    #     support_som = support_feature.cpu().numpy()
    #     som = minisom.MiniSom(2,2,640)
    #     som.train(data_som,2000)
    #
    #     if n == 5:
    #         data_shot = support_feature.reshape(5, 5, 640)
    #         support_feature = torch.mean(data_shot, dim=0)
    #
    #     query_feature = F.normalize(query_feature, dim=1)
    #     support_feature = F.normalize(support_feature, dim=1)
    #     score = torch.matmul(query_feature, support_feature.T)
    #     for i in range(len(query_som)):
    #         temp = [0,0,0,0,0]
    #         t = som.winner(query_som[i])
    #         for k in range(n):
    #             for j in range(5):
    #                 if(t==som.winner(support_som[j+5*k])):
    #                     temp[j]+=1
    #
    #         for j in range(5):
    #             #if(temp[j]<min([4,max(temp)])):
    #             if (temp[j] < 4):
    #                 score[i][j]=0
    #
    #     pred = torch.argmax(score, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return acc
    #
    # def evaluate_net(self, data_query, data_shot, label,n,net):
    #     query_feature1 = self.model(data_query)
    #     support_feature1 = self.model(data_shot)
    #     support_feature,query_feature = net(support_feature1,query_feature1)
    #
    #
    #     # if n == 5:
    #     #     data_shot = support_feature.reshape(5, 5, 640)
    #     #     support_feature = torch.mean(data_shot, dim=0)
    #     #     data_shot1 = support_feature1.reshape(5, 5, 640)
    #     #     support_feature1 = torch.mean(data_shot1, dim=0)
    #
    #     query_feature = F.normalize(query_feature, dim=1)
    #     support_feature = F.normalize(support_feature, dim=1)
    #     score = torch.matmul(query_feature, support_feature.T)
    #     #print("score:",score)
    #     pred = torch.argmax(score, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #
    #     query_feature1 = F.normalize(query_feature1, dim=1)
    #     support_feature1 = F.normalize(support_feature1, dim=1)
    #     score1 = torch.matmul(query_feature1, support_feature1.T)
    #     # print("score:",score)
    #     pred1 = torch.argmax(score1, dim=1)
    #     acc1 = (pred1 == label).type(torch.cuda.FloatTensor).mean().item()
    #     return acc,acc1
    # def evaluate(self, data_query, data_shot, label,n):
    #     query_feature = self.model(data_query)
    #
    #
    #     support_feature = self.model(data_shot)
    #
    #
    #
    #     if n == 5:
    #         data_shot = support_feature.reshape(5, 5, 640)
    #         support_feature = torch.mean(data_shot, dim=0)
    #
    #     query_feature = F.normalize(query_feature, dim=1)
    #     support_feature = F.normalize(support_feature, dim=1)
    #     score = torch.matmul(query_feature, support_feature.T)
    #     #print("score:",score)
    #     pred = torch.argmax(score, dim=1)
    #     acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
    #     return acc
    def evaluate_Euclidean(self, data_query, data_shot, label):
        query_feature = self.model(data_query).cpu().numpy()
        support_feature = self.model(data_shot).cpu().numpy()
        acc = 0
        predicts = []
        # for j in range(len(support_feature)):
        #     support_norm = np.linalg.norm(support_feature[j])
        #     support_feature[j] = support_feature[j] / support_norm
        #     #print(support_norm)
        #
        # for j in range(len(query_feature)):
        #     query_norm = np.linalg.norm(query_feature[j])
        #     query_feature[j] =  query_feature[j] / query_norm
        #     #print(query_norm)
        for j in range(len(query_feature)):
            distances = []
            for k in range(len(support_feature)):
                distance = ((query_feature[j] - support_feature[k]) ** 2).sum()
                distances.append(distance)
            predict = np.argmin(distances)
            predicts.append(predict)

        for j in range(len(label)):
            if (label[j] == predicts[j]):
                acc += 1
 
        acc = acc / len(label)

        return acc

    def evaluate_test(self, data_query, data_shot, label, base_feature):
        query_feature = self.model(data_query).cpu().numpy()
        support_feature = self.model(data_shot).cpu().numpy()
        # query_feature = F.normalize(query_feature, dim=1).cpu().numpy()
        # support_feature = F.normalize(support_feature, dim=1).cpu().numpy()
        support_feature = support_feature - base_feature
        query_feature = query_feature - base_feature
        #print(support_feature,support_feature.shape)
        #print(query_feature,query_feature.shape)
        # print(base_feature,base_feature.shape)
        #support_feature = support_feature / numpy.linalg.norm(support_feature,2,1)[:,None]
        #query_feature = query_feature / numpy.linalg.norm(query_feature, 2, 1)[:, None]

        for j in range(len(support_feature)):
            support_norm = np.linalg.norm(support_feature[j])
            support_feature[j] = support_feature[j] / support_norm
            #print(support_norm)

        for j in range(len(query_feature)):
            query_norm = np.linalg.norm(query_feature[j])
            query_feature[j] = query_feature[j] / query_norm
            #print(query_norm)


        acc = 0
        predicts = []
        for j in range(len(query_feature)):
            distances = []
            for k in range(len(support_feature)):

                distance = ((query_feature[j] - support_feature[k]) ** 2).sum()
                distances.append(distance)
            predict = np.argmin(distances)
            predicts.append(predict)

        for j in range(len(label)):
            if (label[j] == predicts[j]):
                acc += 1

        acc = acc / len(label)

        return acc

        # svm_model = svm.SVC(C=1, kernel='linear')
        # svm_model.fit(support_feature, [0, 1, 2, 3, 4])
        # query_predict = svm_model.predict(query_feature)
        # for j in range(len(label)):
        #     if(label[j] == query_predict[j]):
        #         acc += 1
        # print(acc/len(label))
        # accs.append(acc/len(label))
    def evaluate_test_gen(self, data_query, data_shot, label, base_feature):
        query_feature = self.model(data_query)
        support_feature = self.model(data_shot)
        query_feature = F.normalize(query_feature, dim=1).cpu().numpy()
        support_feature = F.normalize(support_feature, dim=1).cpu().numpy()
        # average_feature = np.mean(base_feature, axis=0)
        # base_feature = base_feature - average_feature
        # support_feature = support_feature - average_feature
        # query_feature = query_feature - average_feature
        # for j in range(len(base_feature)):
        #     norm = np.linalg.norm(base_feature[j])
        #     base_feature[j] = base_feature[j] / norm
        # print(base_feature.shape)
        # for j in range(len(support_feature)):
        #     support_norm = np.linalg.norm(support_feature[j])
        #     support_feature[j] = support_feature[j] / support_norm
        #
        # for j in range(len(query_feature)):
        #     query_norm = np.linalg.norm(query_feature[j])
        #     query_feature[j] = query_feature[j] / query_norm

        # generate_feature = None
        diff = 1000
        for j in range(len(support_feature)):
            max_feature = self.select_feature(support_feature[j], base_feature)
            reshaped_feature = np.zeros((diff, max_feature.shape[1]))
            samples_indices = np.random.randint(low=0, high=np.shape(max_feature)[0], size=diff)
            # steps = np.random.uniform(0, 0.5, size=diff)
            steps = np.random.uniform(size=diff)

            for k in range(len(steps)):
                reshaped_feature[k] = support_feature[j] - steps[k] * (support_feature[j] - max_feature[samples_indices[k]])

            feature = np.expand_dims(support_feature[j], axis=0)
            reshaped_feature = np.concatenate((reshaped_feature, feature), axis=0)
            support_feature[j] = np.mean(reshaped_feature, axis=0)

        for j in range(len(support_feature)):
            support_norm = np.linalg.norm(support_feature[j])
            query_feature[j] = support_feature[j] / support_norm

        acc = 0
        predicts = []
        for j in range(len(query_feature)):
            distances = []
            for k in range(len(support_feature)):
                distance = ((query_feature[j] - support_feature[k]) ** 2).sum()
                distances.append(distance)
            predict = np.argmin(distances)
            predicts.append(predict)

        for j in range(len(label)):
            if (label[j] == predicts[j]):
                acc += 1

        acc = acc / len(label)
        # support_label = []
        # for i in range(self.test_way):
        #     support_label.extend([i]* (diff + 1))
        # acc = 0
        # svm_model = svm.SVC(C=1, kernel='linear')
        # svm_model.fit(generate_feature, support_label)
        # lr_model = LogisticRegression()
        # lr_model.fit(generate_feature, support_label)
        # query_predict = lr_model.predict(query_feature)
        # for j in range(len(label)):
        #     if(label[j] == query_predict[j]):
        #         acc += 1
        #
        # acc = acc / len(label)
        return acc

    def finetine_loop(self, query, support, base_feature):
        label = torch.arange(self.test_way)
        label = label.type(torch.cuda.LongTensor)
        for p in self.model.parameters():
            p.requires_grad = False

        self.test_fc1 = nn.Linear(self.final_dim, self.final_dim).cuda()
        query_feature = self.model(query)
        support_feature = self.model(support)
        similar_support = self.select_feature(support_feature.mean(dim = 0).detach(), base_feature)
        # affine_feature = support_feature
        self.test_fc2 = nn.Linear(len(similar_support), self.test_way).cuda()
        # self.test_fc1 = nn.Linear(len(similar_support), self.test_way).cuda()
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        similar_length = len(similar_support)
        similar_support_train = similar_support.repeat(len(support_feature), 1, 1)
        for i in range(1000):
            # logits = self.test_fc1(support_feature)
            # print(similar_support.shape, affine_feature.shape)
            affine_feature = self.test_fc1(support_feature)
            affine_feature = affine_feature.unsqueeze(1).repeat(1, similar_length, 1)
            distances = torch.exp(((affine_feature - similar_support_train).pow(2).sum(2, keepdim=False).sqrt())).type(torch.cuda.FloatTensor)
            # print(distances)
            logits = self.test_fc2(distances)
            loss = F.cross_entropy(logits, label)
            # print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        label = torch.arange(self.test_way).repeat(15)
        label = label.type(torch.cuda.LongTensor)

        query_feature = self.test_fc1(query_feature)
        query_feature = query_feature.unsqueeze(1).repeat(1, similar_length, 1)
        print(query_feature.shape, similar_support.shape)
        similar_support_test = similar_support.repeat(len(query_feature), 1, 1)
        distances = torch.exp(((query_feature - similar_support_test).pow(2).sum(2, keepdim=False).sqrt()) / 64).type(torch.cuda.FloatTensor)
        predicts = self.test_fc2(distances)
        predicts = torch.argmax(predicts, dim=1)
        acc = (predicts == label).type(torch.cuda.FloatTensor).mean().item()
        # print(acc)
        return acc

    def select_feature(self, feature, base_feature):
        # feature = feature.cpu().numpy()
        distances = []
        for j in range(len(base_feature)):
            distance = ((feature - base_feature[j]) ** 2).sum()
            distances.append(distance)
        index = np.argsort(distances)
        max_output = base_feature[index[0:5]]
        # output = torch.from_numpy(output)
        return max_output

    def evaluate_eulidean_free_lunch(self, query_feature, support_feature, label):
        for j in range(len(support_feature)):
            support_norm = np.linalg.norm(support_feature[j])
            support_feature[j] = support_feature[j] / support_norm
            #print(support_norm)

        for j in range(len(query_feature)):
            query_norm = np.linalg.norm(query_feature[j])
            query_feature[j] = query_feature[j] / query_norm
            #print(query_norm)
        acc = 0
        predicts = []
        for j in range(len(query_feature)):
            distances = []
            # with open("error_value.txt", "a")as f:
            #     f.write("label:" + str(label[j])+'\n')
            for k in range(len(support_feature)):#5
                distance = ((query_feature[j] - support_feature[k]) ** 2).sum()
                # distance = []
                # for i in range(5):
                #     distance.append(((query_feature[j] - support_feature[5*i+k]) ** 2).sum())
                #     distance.sort()
                # with open("error_value.txt", "a")as f:
                #     f.write("Euclidean:" + "  " + ' '.join(str(i) for i in distance) + '\r\n')
                # distance = sum(distance[:3])
                distances.append(distance)
            predict = np.argmin(distances)
            predicts.append(predict)

        for j in range(len(label)):
            if (label[j] == predicts[j]):
                acc += 1

        acc = acc / len(label)

        return acc

    def test_crc(self,shot,p,query,label):
        for j in range(len(shot)):
            support_norm = np.linalg.norm(shot[j])
            shot[j] = shot[j] / support_norm
            # print(support_norm)

        for j in range(len(query)):
            query_norm = np.linalg.norm(query[j])
            query[j] = query[j] / query_norm

        acc = 0
        predicts = []
        for j in range(len(query)):
            errors = []

            coef = np.dot(p,query[j].T)    #num * 1
            for k in range(5):
                coef_c = coef[k*641:(k+1)*641]
                Dc = shot[k*641:(k+1)*641].T
                error = np.square(np.linalg.norm(query[j].T - np.dot(Dc,coef_c)))/np.sum(np.dot(coef_c,coef_c))
                errors.append(error)
            predict = np.argmin(errors)
            predicts.append(predict)

            with open("error_value.txt","a")as f:
                f.write("label:"+str(j%5)+"  "+' '.join(str(int(i)) for i in errors)+'\r\n')
        for j in range(len(label)):
            if (label[j] == predicts[j]):
                acc += 1
        self.batch_ = self.batch_+1

        acc = acc / len(label)

        return acc

