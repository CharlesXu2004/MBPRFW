import math
import numpy.linalg.linalg
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torchvision import transforms
import os
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from utils import cos
import minisom


class BaseModel(nn.Module):

    def __init__(self, model_func, train_way, test_way):
        super().__init__()
        self.model = model_func
        self.train_way = train_way
        self.test_way = test_way
        self.final_dim = self.model.final_feat_dim
        self.base_temperature = 0.1
        self.batch_ = 0
        
        self.train_fc = nn.Linear(self.final_dim, train_way)
        self.test_fc = nn.Linear(self.final_dim, test_way)
        self.train_w = nn.Sequential(
            nn.Linear(self.final_dim, self.final_dim),
            nn.BatchNorm1d(self.final_dim, eps = 0.001, momentum=0.1, affine=True)
        )
        self.train_w[0].weight.data = torch.from_numpy(np.eye(640, dtype=np.float32))
        
        self.eq = nn.Sequential(
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 640),
            nn.BatchNorm1d(640),
            nn.ReLU(inplace=True),
            nn.Linear(640, 8)
        )

    
    def forward(self, x):
        return self.model(x)


    def train_rotation_crop(self, data_rotation, label, label_aug):
        feature = self.model(data_rotation)

        logits = self.train_fc(feature)
        eq_logit = self.eq(feature)

        eq_loss = F.cross_entropy(eq_logit, label_aug)
        en_loss = F.cross_entropy(logits, label)
        loss = 0.8 * eq_loss + en_loss

        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc


    def train_weight(self, data,label,n):
        feature = self.model(data)
        feature = self.train_w(feature)

        for i in range(n):
            for j in range(i+1,n):
                label = int(label[i]==label[j])
                query_feature = F.normalize(feature[i], dim=1)
                support_feature = F.normalize(feature[j], dim=1)
                score = torch.matmul(query_feature, support_feature.T)
                pred = torch.argmax(score, dim=1)
    
        loss = F.cross_entropy(logits, label)
    
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc


    def train_base(self, data,label):
        feature = self.model(data)

        logits = self.train_fc(feature)
        loss = F.cross_entropy(logits, label)

        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc


    def train_loop_trans(self, x, label,data_trans,outs,label_outs):
        feature = self.model(x)
        feature_trans =self.model(data_trans)
        outs = self.model(outs)
        logits = self.train_fc(feature)
        inv_0 = self.inv_head_1(feature)
        inv = self.inv_head_0(feature_trans)
        outs = self.inv_head_1(outs)
        loss_invs = 0
        for i in range(len(feature)):
            loss_inv = simple_contrstive_loss(label[i],inv_0[i], inv[3*i:3*i+3],outs,label_outs)
            loss_invs = loss_invs + loss_inv
        loss_invs = loss_invs/len(feature)
        loss_ce = F.cross_entropy(logits, label)
        loss = loss_ce + 0.1*loss_invs
        pred = torch.argmax(logits, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return loss, acc


    def train_contrastive(self, x, label, temperature):
        x = torch.cat([x[0].cuda(), x[1].cuda()], dim=0)
        batch_size = x.shape[0]
        feature = self.model(x)
        feature = F.normalize(feature, dim=1)
        label = label.cuda()
        label = label.contiguous().view(-1, 1)
        mask = torch.eq(label, label.T).float().cuda()

        anchor_dot_contrast = torch.div(torch.matmul(feature, feature.T), temperature)
        logits = anchor_dot_contrast
        mask = mask.repeat(2, 2)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).cuda(),
            0
        )

        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        loss = (-(temperature / self.base_temperature) * mean_log_prob_pos).mean()

        return loss


    def cluster_evaluate(self, data ,data_query, data_shot, label,n):
        data_feature = self.model(data)
        data_som = data_feature.cpu().numpy()
        query_feature = self.model(data_query)
        query_som = query_feature.cpu().numpy()
        support_feature = self.model(data_shot)
        support_som = support_feature.cpu().numpy()
        som = minisom.MiniSom(5,5,640)
        # som = minisom.MiniSom(2,2,640)
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
            temp = torch.zeros(5)
            for j in range(5):
                temp[j] = math.pow(t[0] - som.winner(support_som[j])[0], 2) + math.pow(t[1] - som.winner(support_som[j])[1], 2)
            score[i][torch.argmax(temp)]=0
        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc


    def cluster5_evaluate(self, data ,data_query, data_shot, label, n):
        data_feature = self.model(data)
        data_som = data_feature.cpu().numpy()
        query_feature = self.model(data_query)
        query_som = query_feature.cpu().numpy()
        support_feature = self.model(data_shot)
        support_som = support_feature.cpu().numpy()
        som = minisom.MiniSom(5,5,640)
        # som = minisom.MiniSom(2,2,640)
        som.train(data_som,2000)

        if n == 5:
            data_shot = support_feature.reshape(5, 5, 640)
            support_feature = torch.mean(data_shot, dim=0)

        query_feature = F.normalize(query_feature, dim=1)
        support_feature = F.normalize(support_feature, dim=1)
        score = torch.matmul(query_feature, support_feature.T)
        for i in range(len(query_som)):
            temp = [0,0,0,0,0]
            t = som.winner(query_som[i])
            for k in range(n):
                for j in range(5):
                    if (math.pow(t[0]-som.winner(support_som[j + 5 * k])[0],2)+math.pow(t[1]-som.winner(support_som[j + 5 * k])[1],2)<64):
                        temp[j] += 1
            for j in range(5):
                if (temp[j]==0):
                    score[i][j]=0
            if(min(temp)<4):
                score[i][temp.index(min(temp))]=0

        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc


    def cluster5_evaluate_end(self, data ,data_query, data_shot, label, n):
        data_feature = self.model(data)
        data_feature = F.normalize(data_feature, dim=1)

        query_feature = self.model(data_query)
        support_feature = self.model(data_shot)
        query_feature = F.normalize(query_feature, dim=1)
        support_feature = F.normalize(support_feature, dim=1)

        data_som = data_feature.cpu().numpy()
        query_som = query_feature.cpu().numpy()
        support_som = support_feature.cpu().numpy()
        som = minisom.MiniSom(5,5,640)
        som.train(data_som,2000)

        if n == 5:
            data_shot = support_feature.reshape(5, 5, 640)
            support_feature = torch.mean(data_shot, dim=0)

        score = torch.matmul(query_feature, support_feature.T)
        score_base = torch.clone(score)
        for i in range(len(data_query)):
            temp = torch.zeros(n*5)
            t = som.winner(query_som[i])
            for k in range(n):
                for j in range(5):
                    temp[j + 5 * k] = math.pow(t[0]-som.winner(support_som[j + 5 * k])[0],2)+math.pow(t[1]-som.winner(support_som[j + 5 * k])[1],2)
            flag = torch.argmax(temp)
            temp[flag] = 0
            score[i][flag%5]=0
            # if(temp[torch.argmax(temp)]>9):
            score[i][torch.argmax(temp)%5]=0

        pred = torch.argmax(score, dim=1)
        pred_base = torch.argmax(score_base, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        acc_base = (pred_base == label).type(torch.cuda.FloatTensor).mean().item()
        return acc,acc_base


    def cluster5_evaluate_protype(self, data ,data_query, data_support, label, n):
        data_feature = self.model(data)
        data_feature = F.normalize(data_feature, dim=1)

        query_feature = self.model(data_query)
        support_feature_ = self.model(data_support)
        query_feature = F.normalize(query_feature, dim=1)
        support_feature = F.normalize(support_feature_, dim=1)

        if n == 5:
            data_shot = support_feature_.reshape(5, 5, 640)
            protype_base = torch.mean(data_shot, dim=0)
            protype = F.normalize(protype_base, dim=1)

        score_base = torch.matmul(query_feature, protype.T)
        pred_base = torch.argmax(score_base, dim=1)
        acc_base = (pred_base == label).type(torch.cuda.FloatTensor).mean().item()
        acc=0

        # data_som = data_feature.cpu().numpy()
        # # query_som = query_feature.cpu().numpy()
        # support_som = support_feature.cpu().numpy()
        # protype_som = protype.cpu().numpy()
        # som = minisom.MiniSom(5, 5, 640)
        # som.train(data_som, 2000)
        temp = torch.zeros(n * 5)
        # for i in range(len(support_feature)):
        #     t = som.winner(support_som[i])
        #     temp[i] = math.pow(t[0]-som.winner(protype_som[i%5])[0],2)+math.pow(t[1]-som.winner(protype_som[i%5])[1],2)
        for i in range(len(support_feature)):
            temp[i] = ((support_feature[i] - protype[i%5]) ** 2).sum()

        flag = torch.argmax(temp)
        temp[flag] = 0
        flag1 = torch.argmax(temp)

        temp_score = torch.matmul(support_feature, protype.T)
        temp_pred = torch.argmax(temp_score, dim=1)

        if temp_pred[flag]!=flag%5:
            temp_support = (protype_base[flag%5]).unsqueeze(0).cuda()
            for k in range(n):
                if flag == k*5+flag%5:
                    pass
                else:
                    temp_support = torch.cat((temp_support,(support_feature_[k*5+flag%5]).unsqueeze(0)),0)
            temp_support = temp_support[1:]
            protype_base[flag%5] = torch.mean(temp_support,dim=0)

        if temp_pred[flag1]!=flag1%5:
            temp_support = (protype_base[flag1%5]).unsqueeze(0).cuda()
            for k in range(n):
                if flag == k*5+flag1%5:
                    pass
                else:
                    temp_support = torch.cat((temp_support,(support_feature_[k*5+flag1%5]).unsqueeze(0)),0)
            temp_support = temp_support[1:]
            protype_base[flag1%5] = torch.mean(temp_support,dim=0)
        protype = F.normalize(protype_base, dim=1)
        score = torch.matmul(query_feature, protype.T)
        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()

        return acc, acc_base


    def evaluate(self, data_query, data_shot, label,n):
        query_feature = self.model(data_query)

        support_feature = self.model(data_shot)

        if n == 5:
            data_shot = support_feature.reshape(5, 5, 640)
            support_feature = torch.mean(data_shot, dim=0)

        query_feature = F.normalize(query_feature, dim=1)
        support_feature = F.normalize(support_feature, dim=1)
        score = torch.matmul(query_feature, support_feature.T)
        pred = torch.argmax(score, dim=1)
        acc = (pred == label).type(torch.cuda.FloatTensor).mean().item()
        return acc


    def evaluate_Euclidean(self, data_query, data_shot, label):
        query_feature = self.model(data_query).cpu().numpy()
        support_feature = self.model(data_shot).cpu().numpy()
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


    def evaluate_test(self, data_query, data_shot, label, base_feature):
        query_feature = self.model(data_query).cpu().numpy()
        support_feature = self.model(data_shot).cpu().numpy()
        support_feature = support_feature - base_feature
        query_feature = query_feature - base_feature

        for j in range(len(support_feature)):
            support_norm = np.linalg.norm(support_feature[j])
            support_feature[j] = support_feature[j] / support_norm

        for j in range(len(query_feature)):
            query_norm = np.linalg.norm(query_feature[j])
            query_feature[j] = query_feature[j] / query_norm

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


    def evaluate_test_gen(self, data_query, data_shot, label, base_feature):
        query_feature = self.model(data_query)
        support_feature = self.model(data_shot)
        query_feature = F.normalize(query_feature, dim=1).cpu().numpy()
        support_feature = F.normalize(support_feature, dim=1).cpu().numpy()
        diff = 1000
        for j in range(len(support_feature)):
            max_feature = self.select_feature(support_feature[j], base_feature)
            reshaped_feature = np.zeros((diff, max_feature.shape[1]))
            samples_indices = np.random.randint(low=0, high=np.shape(max_feature)[0], size=diff)
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
        return acc


    def finetune_loop(self, query, support, base_feature):
        label = torch.arange(self.test_way)
        label = label.type(torch.cuda.LongTensor)
        for p in self.model.parameters():
            p.requires_grad = False

        self.test_fc1 = nn.Linear(self.final_dim, self.final_dim).cuda()
        query_feature = self.model(query)
        support_feature = self.model(support)
        similar_support = self.select_feature(support_feature.mean(dim = 0).detach(), base_feature)
        self.test_fc2 = nn.Linear(len(similar_support), self.test_way).cuda()
        optimizer = torch.optim.SGD(self.parameters(), lr = 0.01, momentum=0.9, dampening=0.9, weight_decay=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        similar_length = len(similar_support)
        similar_support_train = similar_support.repeat(len(support_feature), 1, 1)
        for i in range(1000):
            affine_feature = self.test_fc1(support_feature)
            affine_feature = affine_feature.unsqueeze(1).repeat(1, similar_length, 1)
            distances = torch.exp(((affine_feature - similar_support_train).pow(2).sum(2, keepdim=False).sqrt())).type(torch.cuda.FloatTensor)
            logits = self.test_fc2(distances)
            loss = F.cross_entropy(logits, label)
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
        return acc


    def select_feature(self, feature, base_feature):
        distances = []
        for j in range(len(base_feature)):
            distance = ((feature - base_feature[j]) ** 2).sum()
            distances.append(distance)
        index = np.argsort(distances)
        max_output = base_feature[index[0:5]]
        return max_output


    def evaluate_eulidean_free_lunch(self, query_feature, support_feature, label):
        for j in range(len(support_feature)):
            support_norm = np.linalg.norm(support_feature[j])
            support_feature[j] = support_feature[j] / support_norm

        for j in range(len(query_feature)):
            query_norm = np.linalg.norm(query_feature[j])
            query_feature[j] = query_feature[j] / query_norm
        
        acc = 0
        predicts = []
        for j in range(len(query_feature)):
            distances = []
            for k in range(len(support_feature)):
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

        for j in range(len(query)):
            query_norm = np.linalg.norm(query[j])
            query[j] = query[j] / query_norm

        acc = 0
        predicts = []
        for j in range(len(query)):
            errors = []

            coef = np.dot(p,query[j].T) 
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


def simple_contrstive_loss(label,vi_batch, vi_t_batch, mn_arr,label_outs, temp_parameter=0.1):

    # Define constant eps to ensure training is not impacted if norm of any image rep is zero
    eps = 1e-6
    mn_arrs = None
    flag = 0
    for i in range(len(label_outs)):
        if label==label_outs[i]:
            continue
        else:
            if flag==1:
                mn_arrs = torch.cat((mn_arrs,mn_arr[i].unsqueeze(0)),0)

            else:
                mn_arrs = mn_arr[i].unsqueeze(0)
                flag =1

    vi_batch = vi_batch.unsqueeze(0)
    # L2 normalize vi, vi_t and memory bank representations
    vi_norm_arr = torch.norm(vi_batch, dim=1, keepdim=True)
    vi_t_norm_arr = torch.norm(vi_t_batch, dim=1, keepdim=True)
    mn_norm_arr = torch.norm(mn_arrs, dim=1, keepdim=True)

    vi_batch = vi_batch / (vi_norm_arr + eps)
    vi_t_batch = vi_t_batch/ (vi_t_norm_arr + eps)
    mn_arrs = mn_arrs / (mn_norm_arr + eps)

    print(mn_arrs.shape)
    # Find cosine similarities
    sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_t_mn_mat = (vi_t_batch @ mn_arrs.t())

    # Fine exponentiation of similarity arrays
    exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = torch.sum(exp_sim_vi_t_mn_mat, 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr + eps)

    neg_log_img_pair_probs = -1 * torch.log(batch_prob_arr)
    loss_i_i_t = torch.sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.size()[0]
    return loss_i_i_t