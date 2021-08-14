"""
Created on Wed Oct 21 21:10:24 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import time
import torch
import numpy as np
import sys
sys.path.append('./utils/')
import d2lzh_pytorch as d2l
from torch.nn import Softmax
import transformers
def evaluate_accuracy(data_iter, net, loss, device):
    sftmax = Softmax(dim=1)
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)
            #print('y',y)
            net.eval() 
            y_hat = net(X)
            l = loss(y_hat, y.long())
            y_hat_f = sftmax(y_hat)
            #print('y_hat_f', y_hat_f.argmax(dim=1).float())
            acc_sum += (y_hat_f.argmax(dim=1).float() == y.to(device)).float().sum().cpu().item()
            test_l_sum += l
            test_num += 1
            net.train() 
            n += y.shape[0]
    return [acc_sum / n, test_l_sum]

def train(net, train_iter, valida_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=20):
    loss_list = [100]
    early_epoch = 0
    best_acc = 0
    sftmax = Softmax(dim=1)
    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15 , eta_min=0.0, last_epoch=-1)
    #num_training_steps = len(train_iter)*epochs
    #lr_adjust = transformers.get_cosine_schedule_with_warmup(
    #optimizer,  0.0, num_training_steps,  0.5,  -1)
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            #print('y',y)
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y.long())
            #print(y_hat)
            
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            y_hat_f = sftmax(y_hat)
            #y_hat_f = y_hat_f.type(torch.float64)
            #print('y_', y_hat_f.argmax(dim=1).float())
            train_acc_sum += (y_hat_f.argmax(dim=1).float() == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step()
        current_lr = lr_adjust.get_lr()
        print('current_lr before step', current_lr)
        #lr_adjust.step()
        #current_lr = lr_adjust.get_lr()
        #print('current_lr after step', current_lr)
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        tr_acc, tr_loss = evaluate_accuracy(train_iter, net, loss, device)
        
        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
                % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, valida_loss, valida_acc, time.time() - time_epoch))

        PATH = "./net_DBA.pt"

        if valida_acc > best_acc:
            best_acc = valida_acc
            print('best_acc', best_acc)
            torch.save(net.state_dict(), PATH)
            #lr_adjust.step()
        #else:
            #net.load_state_dict(torch.load(PATH))


        #if early_stopping and loss_list[-2] < loss_list[-1]:
            #if early_epoch == 0:
                #torch.save(net.state_dict(), PATH)
            #early_epoch += 1
            #loss_list[-1] = loss_list[-2]
            #if early_epoch == early_num:
                #net.load_state_dict(torch.load(PATH))
                #break
        #else:
            #early_epoch = 0
            
    net.load_state_dict(torch.load(PATH))
    
    #d2l.set_figsize()
    #d2l.plt.figure(figsize=(8, 8.5))
    #train_accuracy = d2l.plt.subplot(221)
    #train_accuracy.set_title('train_accuracy')
    #d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    #d2l.plt.xlabel('epoch')
    #d2l.plt.ylabel('train_accuracy')

    #test_accuracy = d2l.plt.subplot(222)
    #test_accuracy.set_title('valida_accuracy')
    #d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    #d2l.plt.xlabel('epoch')
    #d2l.plt.ylabel('test_accuracy')

    #loss_sum = d2l.plt.subplot(223)
    #loss_sum.set_title('train_loss')
    #d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='red')
    #d2l.plt.xlabel('epoch')
    #d2l.plt.ylabel('train loss')

    #test_loss = d2l.plt.subplot(224)
    #test_loss.set_title('valida_loss')
    #d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    #d2l.plt.xlabel('epoch')
    #d2l.plt.ylabel('valida loss')

    #d2l.plt.show()
    #print('epoch %d, loss %.4f, train acc %.3f, time %.1f sec'
            #% (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, time.time() - start))
