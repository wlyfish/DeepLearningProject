# -*- encoding : utf-8 -*-
"""
@project = DeepLearningProject
@file = config
@author = wly
@create_time = 2022/9/24 16:32
"""
# banknote classification config

# 超参配置


class Hyperparameter:
    # ################################################################
    #                             Data
    # ################################################################
    device = 'cuda'  # cuda
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
    batch_size = 64     # 一次取64条数据进行运算
    init_lr = 1e-3  # 初始学习率
    epochs = 100    # 训练100轮
    verbose_step = 10  # 日志打印间隔
    save_step = 200  # 模型保存间隔


HP = Hyperparameter()
