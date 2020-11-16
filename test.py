#coding:utf8
import ipdb
import os
import models
import torch as t
import scipy.io as sio
import numpy as np

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from config import opt
from data.dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torchnet import meter
from utils.visualize import Visualizer

import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def test(**kwargs):
    opt._parse(kwargs)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.test_model_path:
        model.load(opt.test_model_path)
    else:
        raise

    model.to(opt.device)

    # step2: data
    test_data = Test_TextData(opt.test_image_path)
    test_dataloader = DataLoader(test_data, 1,
                        shuffle=False,num_workers=opt.num_workers)
    
    softmax = t.nn.Softmax(dim=1)   

    # test
    model_index = opt.test_model_path.split('.')[-2].split('_')[-1]
    f = open('result_auc_{}.txt'.format(model_index), 'w')

    for ii,(data, filename) in tqdm(enumerate(test_dataloader)):
        # train model 
        input = data.to(opt.device)
        score = model(input)
        p = softmax(score)

        np_p = p.detach().cpu().numpy()
        result_line = filename[0] + ' ' + str(np.argmax(np_p)) + ' ' + str(np_p[0][np.argmax(np_p)])
        print(result_line)

        f.writelines(result_line + '\n')
        print(ii)   
        #print(score.detach().cpu().numpy())   
        # meters update and visualize
    f.close()

def val(**kwargs):
    opt._parse(kwargs)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.test_model_path:
        model.load(opt.test_model_path)
    else:
        raise

    model.to(opt.device)

    # step2: data
    test_data = TextData(opt.data_root, opt.test_txt_path)
    test_dataloader = DataLoader(test_data, 1,
                        shuffle=False,num_workers=opt.num_workers)
    
    softmax = t.nn.Softmax(dim=1)   

    # test
    model_index = opt.test_model_path.split('.')[-2].split('_')[-1]
    
    y = []
    pred = []
    right_num = []
    for ii,(data, label) in tqdm(enumerate(test_dataloader)):
        # train model 
        input = data.to(opt.device)
        score = model(input)
        p = softmax(score)
        
        np_p = p.detach().cpu().numpy()

        y.append(np.argmax(np_p))
        pred.append(np_p[0][np.argmax(np_p)])

        #ipdb.set_trace()
        right_num.append(int(np.argmax(np_p) == label.cpu().numpy()))
        print(ii)   
        #print(score.detach().cpu().numpy())   
        # meters update and visualize
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    print(auc(fpr, tpr))
    print(np.mean(np.array(right_num)))
    ipdb.set_trace()

def group_val(**kwargs):
    opt._parse(kwargs)
  
    auc_ = 0
    acc_ = 0
    auc_index = 0
    acc_index = 0
    for index in range(0, 73, 2):
        # step1: configure model
        model = getattr(models, opt.model)()
        if opt.test_model_path:
            model.load("./checkpoints/text_new_" + str(index) + ".pth")
            #model.load(opt.test_model_path)
        else:
            raise

        model.to(opt.device)

        # step2: data
        test_data = TextData(opt.data_root, opt.test_txt_path)
        test_dataloader = DataLoader(test_data, 1,
                            shuffle=False,num_workers=opt.num_workers)
        
        softmax = t.nn.Softmax(dim=1)   

        # test
        model_index = opt.test_model_path.split('.')[-2].split('_')[-1]
        
        y = []
        pred = []
        right_num = []


        for ii,(data, label) in tqdm(enumerate(test_dataloader)):
            # train model 
            input = data.to(opt.device)
            score = model(input)
            p = softmax(score)
            
            np_p = p.detach().cpu().numpy()

            y.append(np.argmax(np_p))
            pred.append(np_p[0][0])

            #ipdb.set_trace()
            right_num.append(int(np.argmax(np_p) == label.cpu().numpy()))
            #print(score.detach().cpu().numpy())   
            # meters update and visualize
        
        if np.sum(y) == 0 or np.sum(y) == 552:
            continue

        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        new_auc = auc(fpr, tpr)
        new_acc = np.mean(np.array(right_num))
        print(auc(fpr, tpr))
        print(np.mean(np.array(right_num)))
        #ipdb.set_trace()
        (auc_, auc_index) = (auc_, auc_index) if auc_ > new_auc else (new_auc, index)
        (acc_, acc_index) = (acc_, acc_index) if acc_ > new_acc else (new_acc, index)

        print("The highest auc: {}, model index: {}".format(auc_, auc_index))
        print("The highest acc: {}, model index: {}".format(acc_, acc_index))

def group_test(**kwargs):
    auc = 0
    acc = 0
    auc_index = 0
    acc_index = 0
    for i in range(0, 72, 2):
        new_auc, new_acc = group_val(i)
        auc, auc_index = auc, auc_index if auc > new_auc else new_auc, i
        acc, acc_index = acc, acc_index if acc > new_acc else new_acc, i
    print("The highest auc: {}, model index: {}".format(auc, auc_index))
    print("The highest acc: {}, model index: {}".format(acc, acc_index))

if __name__=='__main__':
    val()

