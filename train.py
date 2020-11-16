#coding:utf8
import ipdb
import os
import models
import torch as t
import scipy.io as sio

from config import opt
from data.dataset import *
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from torchnet import meter
from utils.visualize import Visualizer

import scipy.io
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_device_index

def weight_init(m):
    if isinstance(m, t.nn.Conv2d):
        t.nn.init.kaiming_uniform(m.weight)

def train(**kwargs):
    opt._parse(kwargs)
    vis = Visualizer(opt.env,port = opt.vis_port)

    # step1: configure model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load_new(opt.load_model_path)
    else:
        print('Initialize the model!')
        model.apply(weight_init)

    model.to(opt.device)

    # step2: data
    train_data = TextData(opt.data_root, opt.train_txt_path)
    val_data = TextData(opt.data_root, opt.val_txt_path)
    train_dataloader = DataLoader(train_data,opt.batch_size,
                        shuffle=True,num_workers=opt.num_workers)
    val_dataloader = DataLoader(val_data,opt.batch_size,
                        shuffle=False,num_workers=opt.num_workers)

    # step3: criterion and optimizer
    criterion = t.nn.CrossEntropyLoss()
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: meters
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)
    previous_loss = 1e10
    
    # train
    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        confusion_matrix.reset()

        for ii,(data,label) in tqdm(enumerate(train_dataloader)):
            # train model 
            input = data.to(opt.device)
            target = label.to(opt.device)
            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target) 
            loss.backward()
            #for n, p in model.named_parameters(): 
            #    print(n)
            #    h = p.register_hook(lambda grad: print(grad)) 
            optimizer.step()
            
            # meters update and visualize
            loss_meter.add(loss.item())
            confusion_matrix.add(score.data, target.data)
            if ii % opt.print_freq == 0:
                vis.plot('loss', loss_meter.value()[0])
                
                # enter debug mode
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
            if ii % (opt.print_freq * 10) == 0:
                vis.images(input.cpu().numpy(), opts=dict(title='Label', caption='Label'), win=1)
                print('Epoch: {} Iter: {} Loss: {}'.format(epoch, ii, loss))
        
        if epoch % 2 == 0:
            model.save('./checkpoints/' + opt.env + '_' + str(epoch) + '.pth')
        
        # validate and visualize
        val_cm,val_accuracy = val(model,val_dataloader)

        vis.plot('val_accuracy',val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(epoch = epoch,loss = loss_meter.value()[0],val_cm = str(val_cm.value()),train_cm=str(confusion_matrix.value()),lr=lr))
        train_cm = confusion_matrix.value()
        t_accuracy = 100. * (train_cm[0][0] + train_cm[1][1]) / (train_cm.sum())
        vis.plot('train_accuracy',t_accuracy)
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay        
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

def val(model,dataloader):
    model.eval()
    confusion_matrix = meter.ConfusionMeter(2)
    for ii, data in tqdm(enumerate(dataloader)):
        input, label = data
        val_input = Variable(input, volatile=True)
        if opt.use_gpu:
            val_input = val_input.cuda()
        score = model(val_input)
        confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

    model.train()
    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy

if __name__=='__main__':
    train()

