# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = config
@author = wly
@create_time = 2022/9/24 16:32
"""
# banknote classification config

# 超参配置
# yaml


class Hyperparameter:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cpu'  # cuda
    data_dir = './data/'
    data_path = './data/data_banknote_authentication.txt'
    trainset_path = './data/train.txt'
    devset_path = './data/dev.txt'
    testset_path = './data/test.txt'

    in_features = 4  # input feature dim
    out_dim = 2  # output feature dim (classes number)
    seed = 1234  # random seed

    # ################################################################
    #                             Model Structure
    # ################################################################
    layer_list = [in_features, 64, 128, 64, out_dim]
    # ################################################################
    #                             Experiment
    # ################################################################
    batch_size = 64
    init_lr = 1e-3
    epochs = 100
    verbose_step = 10
    save_step = 200


HP = Hyperparameter()
