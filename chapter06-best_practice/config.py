# coding:utf8
import warnings
import torch as t

def setup_log():
    import logging
    import os
    import datetime

    # Initialize logging
    # simple_format = '%(levelname)s >>> %(message)s'
    medium_format = (
        '%(levelname)s : %(filename)s[%(lineno)d]'
        ' >>> %(message)s'
    )
    log_dir = os.path.join(os.path.dirname(__file__), 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_file = os.path.join(log_dir, 'chapter06.log')

    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format=medium_format
    )
    logging.info('@{} created at {}'.format(
        log_file,
        datetime.datetime.now())
    )

setup_log()

class DefaultConfig(object):
    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口
    # model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    model = 'ResNet34'

    # train_data_root = './data/train/'  # 训练集存放路径
    train_data_root = '/disk4t0/PyTorchDev_data/dogs-vs-cats/train/'
    # test_data_root = './data/test1'  # 测试集存放路径
    test_data_root = '/disk4t0/PyTorchDev_data/dogs-vs-cats/test1'
    # load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    load_model_path = './checkpoints/resnet34_0301_15:01:01.pth'

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    # max_epoch = 10
    max_epoch = 3
    # lr = 0.001  # initial learning rate
    lr = 0.005
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数


    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        opt.device =t.device('cuda') if opt.use_gpu else t.device('cpu')


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
