"""
Created on Wed Oct 21 21:10:24 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
import h5py
import torch.utils.data as Data

import extract_samll_cubic

def load_dataset(Dataset):
    if Dataset == 'PC':
        uPavia = sio.loadmat('./dataset/PaviaU.mat')
        gt_uPavia = sio.loadmat('./dataset/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 207400
        #VALIDATION_SPLIT = 0.999
        VALIDATION_SPLIT = 0.3

        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'IP':
        uPavia = sio.loadmat('./dataset/Indian_pines_corrected.mat')
        gt_uPavia = sio.loadmat('./dataset/Indian_pines_gt.mat')
        data_hsi = uPavia['indian_pines_corrected']
        gt_hsi = gt_uPavia['indian_pines_gt']
        TOTAL_SIZE = 21025
        #VALIDATION_SPLIT = 0.999
        VALIDATION_SPLIT = 0.1

        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)        
    if Dataset == 'DFC2013':
        HS = sio.loadmat('./dataset/DFC2013_Houston.mat')
        gt_HS = sio.loadmat('./dataset/DFC2013_Houston_gt.mat')
        data_hsi = HS['HSI'].astype(np.float32)
        gt_hsi = gt_HS['gt']
        TOTAL_SIZE = 15029
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    
    if Dataset == 'DN':
        DN = sio.loadmat('../datasets/Dioni.mat')
        gt_DN = sio.loadmat('../datasets/Dioni_gt.mat')
        data_hsi = DN['Dioni']
        gt_hsi = gt_DN['Dioni_gt']
        TOTAL_SIZE = 20024
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'DFC2018':
        DFC = sio.loadmat('../datasets/DFC2018_Houston.mat')
        gt_DFC = sio.loadmat('../datasets/DFC2018_Houston_gt.mat')
        data_hsi = DFC['DFC2018_Houston']
        gt_hsi = gt_DFC['DFC2018_Houston_gt']
        TOTAL_SIZE = 504712
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    if Dataset == 'KSC':
        DFC = sio.loadmat('./dataset/KSC.mat')
        gt_DFC = sio.loadmat('./dataset/KSC_gt.mat')
        data_hsi = DFC['KSC']
        gt_hsi = gt_DFC['KSC_gt']
        TOTAL_SIZE = 504712
        VALIDATION_SPLIT = 0.9
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT
def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()

def sampling1(samples_num, ground_truth):
    train = {}
    test = {}
    val = {}
    labels_loc = {}
    m = max(ground_truth)
    #print('max(ground_truth)', m )
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        nb_val = min(samples_num,len(indexes))
        print('len(indexes)', len(indexes))
        train[i] = indexes[:nb_val]
        val[i] = indexes[nb_val:2*nb_val]
        test[i] = indexes[2*nb_val:]
        #print('t_index', t_index)
        #test[i] = indexes[nb_val:]
    val_indexes = []
    train_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        val_indexes += val[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, val_indexes, test_indexes

def sampling2(ground_truth, proportion):
    train = {}
    test = {}
    val = {}
    labels_loc = {}
    m = max(ground_truth)
    for i in range(m):
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indexes)
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3)
        else:
            nb_val = 0
        print('nb_val', nb_val)
        train[i] = indexes[:nb_val]
        val[i] = indexes[nb_val:2*nb_val]
        test[i] = indexes[2*nb_val:]
    train_indexes = []
    val_indexes = []
    test_indexes = []
    for i in range(m):
        train_indexes += train[i]
        val_indexes += val[i]
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes,val_indexes, test_indexes

def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([100, 255, 100])/255.
        if item == 3:
            y[index] = np.array([0,0,255])/255.
        if item == 4:
            y[index] = np.array([255, 255, 0])/255.
        if item == 5:
            y[index] = np.array([255, 0, 255])/255.
        if item == 6:
            y[index] = np.array([255, 100, 100])/255.
        if item == 7:
            y[index] = np.array([150, 75, 255])/255.
        if item == 8:
            y[index] = np.array([150, 75, 75])/255.
        if item == 9:
            y[index] = np.array([100, 100, 255])/255.
        if item == 10:
            y[index] = np.array([0, 200, 200])/255.
        if item == 11:
            y[index] = np.array([0, 100, 100])/255.
        if item == 12:
            y[index] = np.array([100, 0, 100])/255.
        if item == 13:
            y[index] = np.array([128, 128, 0])/255.
        if item == 14:
            y[index] = np.array([200, 100, 0])/255.
        if item == 15:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 16:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 17:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 18:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 19:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 20:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 21:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 22:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 23:
            y[index] = np.array([0, 255, 0]) / 255.
    return y


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):

    gt_all = gt[total_indices] - 1
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    #all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      #PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION)

    #x_val = x_test_all[-VAL_SIZE:]
    #y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all
    y_test = y_test
    #x_test = x_test_all[:-VAL_SIZE]
    #y_test = y_test[:-VAL_SIZE]
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    #x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    #y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    #print('y1_tensor_valida',y1_tensor_valida.shape)
    #print('x1_tensor_valida',x1_tensor_valida.shape)
    #torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test,y1_tensor_test)

    #all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    #all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    #all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    #torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)


    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    #valiada_iter = Data.DataLoader(
        #dataset=torch_dataset_valida,
        #batch_size=batch_size,
        #shuffle=True,
        #num_workers=0,
    #)
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    #all_iter = Data.DataLoader(
        #dataset=torch_dataset_all,
        #batch_size=batch_size,
        #shuffle=False,
        #num_workers=0,
    #)
    return train_iter, test_iter#, test_iter, all_iter #, y_test

def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []
    for X, y in all_iter:
        X = X.to(device)
        net.eval()
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(pred_test)):
        pred_test[i] = pred_test[i] + 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    path = './'
    classification_map(y_re, gt_hsi, 600,
                       path + '/classification_maps/' + Dataset + '_' + net.name +  '.eps')
    classification_map(gt_re, gt_hsi, 600,
                       path + '/classification_maps/' + Dataset + '_gt.eps')
    print('------Get classification maps successful-------')
