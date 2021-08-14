"""
Created on Fri Oct 23 20:58:38 2020

@author: xuegeeker
@email: xuegeeker@163.com
"""

import numpy as np
import time
import collections
from torch import optim
import torch
from sklearn import metrics, preprocessing
import datetime
import numpy as np
import sys
sys.path.append('./networks')
import network
import train
from sklearn.decomposition import PCA
#import time
sys.path.append('./utils')
from generate_pic import aa_and_each_accuracy, sampling1, sampling2, load_dataset, generate_png, generate_iter
import record, extract_samll_cubic
from record import print_confusion_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cuda:0')
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')

print('-----Importing Dataset-----')
global Dataset
dataset = 'DFC2013'
Dataset = dataset.upper()
#data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE,VALIDATION_SPLIT = load_dataset(Dataset)
data_hsi, gt_hsi, class_names = load_dataset(Dataset)


#data_hsi,pca = applyPCA(data_hsi,numComponents=30)
print(data_hsi.shape)

ROWS, COLUMNS, BAND = data_hsi.shape
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
print('data', data.shape)



gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]),)
print('gt',gt.shape)
CLASSES_NUM = int(max(gt))
print('The class numbers of the HSI data is:', CLASSES_NUM)




##########################################

print('-----Importing Setting Parameters-----')
ITER = 5
PATCH_LENGTH = 2
lr, num_epochs, batch_size = 0.01, 50, 128
loss = torch.nn.CrossEntropyLoss()

img_rows = 2*PATCH_LENGTH+1
img_cols = 2*PATCH_LENGTH+1
INPUT_DIMENSION = data_hsi.shape[2]
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]

ratio = 0.4

print('ALL_SIZE',ALL_SIZE)
VAL_SIZE = int(ratio*ALL_SIZE)
TRAIN_SIZE = int(ratio*ALL_SIZE)
TEST_SIZE = ALL_SIZE - TRAIN_SIZE
print('ALL_SIZE:',ALL_SIZE, 'TRAIN_SIZE:',TRAIN_SIZE, 'VAL_SIZE:',VAL_SIZE )


SAMPLES_NUM = 40

KAPPA = []
OA = []
AA = []
TRAINING_TIME = []
TESTING_TIME = []
ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

print('part of data before preprocessing',data[100:110][0])
#max_value = np.amax(data)
#min_value = np.amin(data)
#idx =np.where(data == max_value)
#print('max', max_value, 'min',min_value)
data = preprocessing.scale(data)
#data = (data - np.amin(data))/np.ptp(data)

print('part of data after preprocessing',data[0:10][0])
#max_value = np.amax(data)
#min_value = np.amin(data)
#idx =np.where(data == max_value)
#print('max', max_value, 'min',min_value)
whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)

print('padded_data', padded_data[100]) #[100:115][100:115])
for index_iter in range(ITER):
    print('-----Begining to conduct the ' + str(index_iter + 1) + ' iter training process-----')
    net = network.HResNetAM(BAND, CLASSES_NUM)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])
    #train_indices,val_indeces, test_indices = sampling1(SAMPLES_NUM, gt)
    train_indices,val_indeces, test_indices = sampling2(gt, 0.7)
    #print('train_indices', len(train_indices))
    _,_, total_indices = sampling2(gt, 1)
    
    TRAIN_SIZE = len(train_indices)
    print('Train size: ', TRAIN_SIZE)
    VAL_SIZE = len(val_indeces)
    print('Validation size: ', VAL_SIZE)
    #TEST_SIZE = ALL_SIZE - TRAIN_SIZE - VAL_SIZE
    TEST_SIZE = len(test_indices)

    print('Test size: ', TEST_SIZE)
    print('Test size: ' , len(test_indices))
    print('total indeces', len(total_indices))


    
    print('-----Selecting Small Pieces from the Original Cube Data-----')
    #train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, ALL_SIZE, total_indices, VAL_SIZE,
                  #whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    #train_iter, valida_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, ALL_SIZE, total_indices, VAL_SIZE,
                  #whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    train_iter, valida_iter = generate_iter(TRAIN_SIZE, train_indices, VAL_SIZE, val_indeces, ALL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    _, test_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, ALL_SIZE, total_indices, VAL_SIZE,
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
    _, all_iter = generate_iter(TRAIN_SIZE, train_indices, len(total_indices), total_indices, ALL_SIZE, total_indices, VAL_SIZE,
                whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt)
        
    print('-----Begining to Train The Model with Training Dataset-----')
    tic1 = time.time()
    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs)
    toc1 = time.time()
    
    pred_test = []
    tic2 = time.time()
    with torch.no_grad():
        for X, y in test_iter:
            X = X.to(device)
            net.eval()
            y_hat = net(X)
            pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
    toc2 = time.time()
    collections.Counter(pred_test)
    gt_test = gt[test_indices] - 1
    
    overall_acc = metrics.accuracy_score(pred_test, gt_test)
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test)
    fig,df_cm = print_confusion_matrix(confusion_matrix, class_names)
    fig.savefig('confusion_matrix'+'_'+Dataset+'.png',dpi=600)
    df_cm.to_csv('confusion_matrix'+'_'+Dataset+'.csv')
    each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test)
    print(each_acc)
    torch.save(net.state_dict(), "./models/" + Dataset + '.pt')
    KAPPA.append(kappa)
    OA.append(overall_acc)
    AA.append(average_acc)
    TRAINING_TIME.append(toc1 - tic1)
    TESTING_TIME.append(toc2 - tic2)
    ELEMENT_ACC[index_iter, :] = each_acc

    
#print("--------" + net.name + " Training Finished-----------")
record.record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                     './records/' + net.name + 'Patch'+ str(2*PATCH_LENGTH+1) + 'Time' + day_str + '_' + Dataset + 'TrainingSamples' + str(SAMPLES_NUM) + 'lr' + str(lr) + '.txt')

generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)
