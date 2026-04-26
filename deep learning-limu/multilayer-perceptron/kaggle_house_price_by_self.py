from urllib.parse import unquote

from d2l import torch as d2l
import torch
from matplotlib import pyplot as plt
from torch import nn
import hashlib
import os
import numpy as np
import pandas as pd
import requests
import zipfile
from torch.utils import data
import random
DATA_URL = 'https://cloud.solmount.top/'

DATA_HUB = dict()
DATA_HUB['train_data'] = (DATA_URL + 'f/PxpFK/train.csv', '')
DATA_HUB['test_data'] = (DATA_URL + 'f/E3pFe/test.csv', '')


def file_sha1(file_path):
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                break
            sha1.update(data)
            '''直接把需要的值return出去，太妙了'''
    return sha1.hexdigest()


def download(name, cache_dir=os.path.join('..', 'data')):
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)

    file_name = url.split('/')[-1]
    file_path = os.path.join(cache_dir, file_name)

    if os.path.exists(file_path):
        current_sha1 = file_sha1(file_path)
        DATA_HUB[name] = (url, current_sha1)
        if sha1_hash and current_sha1 == sha1_hash:
            print(f'{file_name} already exists, use cached file.')
            return file_path

    print(f'downloading {file_name} from {url}')
    r = requests.get(url, stream=True, verify=True)
    r.raise_for_status()
    with open(file_path, 'wb') as f:
        f.write(r.content)

    current_sha1 = file_sha1(file_path)
    DATA_HUB[name] = (url, current_sha1)
    '''需要更新字典里的值就直接用新值替换就行了'''
    print(f'{file_name} downloaded')
    return file_path


# encode_name = '%E4%B8%9C%E8%A5%BF-%E6%9E%97%E4%BF%8A%E5%91%88%23hfQQh_01'
# decode_name = unquote(encode_name)
# DATA_HUB['decode_name'] = (DATA_URL + encode_name + '.wav', '')
# download('decode_name')
# def download_extract(name,folder=None):
#     file_name = download(name)
#     base_dir = os.path.dirname(file_name)
#     data_dir,ext = os.path.splitext(file_name)
#     if ext == '.zip':
#         file_pointer = zipfile.ZipFile(file_name, 'r')
#     else:
#         assert False, '仅支持解压缩zip文件'
#     file_pointer.extractall(base_dir)
#     return os.path.join(base_dir, folder) if folder else data_dir
def download_extract(name):
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    '''splitext专门拆后缀名'''
    if ext == '.zip':
        file_pointer = zipfile.ZipFile(fname,'r')
    else:
        assert False,'仅支持zip文件的解压'
    file_pointer.extractall(base_dir)
    return base_dir
DATA_HUB['test_extract'] = (DATA_URL + 'f/Kx9He/kaggle_house_price.zip','24e2a2967ef7ed3c5f969ae73845083a68181420')
data_extract_dir = download_extract('test_extract')
print(f'解压后的文件在{data_extract_dir}')

print('----------开始进行数据预处理------------')

train_data = pd.read_csv(os.path.join(data_extract_dir,'train.csv'))
test_data = pd.read_csv(os.path.join(data_extract_dir,'test.csv'))
print('----------开始拼接数据集----------')
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
numeric_features = all_features.dtypes[all_features.dtypes!='object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x-x.mean()) / x.std()
)
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
print('----------将数据转换为tensor张量----------')
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
print('----------开始训练----------')

loss = nn.MSELoss()
in_features = train_features.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
def get_net():
    net = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Linear(256,1)
    )
    return net.to(device)
def log_rmse(net, features, labels):
    '''
    最小值为1
    最大值无穷大，即不做限制
    '''
    was_training = net.training
    net.eval()
    with torch.no_grad():
        features = features.to(device)
        labels = labels.to(device)
        clipped_preds = torch.clamp(net(features),1,float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    if was_training:
        net.train()
    return rmse.item()
def load_array(data_array, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle=is_train,
                           pin_memory=device.type == 'cuda')

def train (net, train_features, train_label, test_features, test_label,
           num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_label), batch_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            l = loss(net(X), y)
            '''
            这里训练和测试使用的是不同的损失函数，使用log_rmse训练会使得计算量增大
            训练的目的是让模型的输出结果和标签尽可能接近
            使用adam损失函数已经足够了
            但给然看的，需要输出的结果，是需要使用相对误差的，以更贴合实际情况
            '''
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_label))
        if test_label is not None:
            test_ls.append(log_rmse(net,test_features, test_label))
    return train_ls, test_ls
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        '''
        下一次循环的时候，会被新的数据覆盖
        但在覆盖之前，有效数据就已经赋值给对应的变量了
        '''
        X_part, y_part = X[idx, :] , y[idx]
        if j == i :
            X_val, y_val = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], dim=0)
            y_train = torch.cat([y_train, y_part], dim=0)
    return X_train, y_train,X_val, y_val
def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls_sum, valid_ls_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        '''
        *data, 
        注意这个操作，太妙了，
        data解包之后正好事四个参数，
        data里的参数名与函数需要传入的不同也并不会有什么影响
        会按顺序一次进行赋值
        '''
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        train_ls_sum += train_ls[-1]
        valid_ls_sum += valid_ls[-1]
        if i == 0:
            plt.figure(figsize=(8,5))
            plt.plot(list(range(1, num_epochs+1)),train_ls,label='train')
            plt.plot(list(range(1, num_epochs+1)),valid_ls, label='test')
            plt.xlabel('epoch')
            plt.ylabel('rmse')
            plt.yscale('log')
            plt.legend()
            plt.grid(True)
            plt.show()
    return train_ls_sum / k, valid_ls_sum / k
k = 5
n_trails = 30
best_valid_l = float('inf')
best_params = None
def find_parms(n_trails):
    for trail in range(n_trails):
        lr = 10 ** random.uniform(-3, -1)
        wd = 10 ** random.uniform(-3, -1)
        bs = random.choice([32, 64, 128, 256, 512])
        epochs = random.choice([100, 500, 1000, 1500])
        _, valid_l = k_fold(k, train_features, train_labels, epochs, lr, wd, bs)
        print(f'[{trail+1}/{n_trails}] lr={lr:.4f}, weight_decays={wd:.4f}, batch_size={bs:.4f}, epochs={epochs:d}, '
              f'--> valid: {valid_l:.4f}')
        if valid_l < best_valid_l:
            best_valid_l = valid_l
            best_params = {'lr': lr, 'wd':wd, 'bs':bs, 'epochs':epochs}
    print(f'最佳参数: {best_params}, valid_rmse={best_valid_l:.4f}')

num_epochs, lr, weight_decay, batch_size = 1000, 0.002, 0.001, 32
# train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
#                           weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
#       f'平均验证log rmse: {float(valid_l):f}')
# 最佳参数: {'lr': 0.0021125412863253878, 'wd': 0.0010985087587355735, 'bs': 32, 'epochs': 1000}, valid_rmse=0.1410
def train_and_pred(train_features, test_features, train_labels, test_data, num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None, num_epochs, lr,
                        weight_decay, batch_size)
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, num_epochs+1), train_ls, label='train')
    plt.xlabel('epoch')
    plt.ylabel('rmse')
    plt.yscale('log')
    plt.xlim(1, num_epochs)
    plt.legend()
    plt.grid(True)
    plt.show()
    print(f'train rmse: {train_ls[-1]:.4f}')

    preds = net(test_features.to(device)).detach().cpu().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)