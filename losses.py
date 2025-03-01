import torch
import torch.nn as nn
import numpy as np

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