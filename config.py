# coding:utf8
import warnings
import torch as t
import numpy as np

class DefaultConfig(object):
    env = 'text_new'  # visdom environment
    vis_port = 8099 # visdom port num
    model = 'ResNet34'# 'WaveletNet'  # model used whose name should consist with the name in 'models/__init__.py'

    data_root = './data/data/'
    train_txt_path = './data/train_image_paths.txt'
    val_txt_path = './data/val_image_paths.txt'
    load_model_path = None #'./checkpoints/ldct63_90.pth'
    test_model_path = './checkpoints/text_new_20.pth'

    test_image_path = './data/test_data'
 
    gpu_device_index = '0'   
    batch_size = 4 # batch size
    val_batch_size = 1
    use_gpu = True  # user GPU or not
    num_workers = 0  # how many workers for loading data
    print_freq = 10  # print info every N batch
    save_model_freq = 2 #how many epoch to save model 

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 401
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 

    def _parse(self, kwargs):
        """
        update parameter in config file
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
