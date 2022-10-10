import numpy as np
import os
import time
import sys
import copy
import torch
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import h5py
import gtsrb_dataset
import argparse
from transfer_easyCNN import CNN
from data_loader import GetLoader

parser = argparse.ArgumentParser()
parser.add_argument("source", type=str, default='gtsrb', help='Dataset to use.')
parser.add_argument("target", type=str, default='cifar', help='Dataset to use.')
parser.add_argument("-p", type=bool, default=False, help='Whether to poison.')
parser.add_argument("-s", type=bool, default=False, help='Whether to poison from scratch or tuning.')
parser.add_argument("-r", type=int, default=0, help='Ratio to poison per batch.')
parser.add_argument("-d", type=int, default=3, help='Size of trigger.')
parser.add_argument("-t", type=int, default=0, help='Attack target label')
parser.add_argument("-e", type=int, default=10, help='Training epoches.')
args = parser.parse_args()

MODEL_ROOT = 'saved_models'
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.cudnn.benchmark = False


LEARNING_RATE = 1e-3
BATCH_SIZE = 128
N_EPOCH = args.e
LOG_INTERVAL = 20
SOURCE_DATASET_NAME = args.source
TARGET_DATASET_NAME = args.target
if SOURCE_DATASET_NAME == 'gtsrb':
    SOURCE_NUM = 43
else:
    SOURCE_NUM = 10
if TARGET_DATASET_NAME == 'gtsrb':
    TARGET_NUM = 43
else:
    TARGET_NUM = 10


def load_data(dataset_name):
    if dataset_name == 'cifar':
        img_transform_cifar = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root='CIFAR',
            train=True,
            transform=img_transform_cifar,
            target_transform=None,
            download=True)

    elif dataset_name == 'gtsrb':
        img_transform_gtrsb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                 (0.2724, 0.2608, 0.2669))
        ])
        dataset = gtsrb_dataset.GTSRB(
            root_dir='./',
            train=True,
            transform=img_transform_gtrsb)

    elif dataset_name == 'mnist':
        img_transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        dataset = datasets.MNIST(
            root='./',
            train=True,
            transform=img_transform_mnist,
            download=True
        )

    elif dataset_name == 'mnistm':
        img_transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        train_list = './mnist_m/mnist_m_train_labels.txt'
        dataset = GetLoader(
            data_root='./mnist_m/mnist_m_train',
            data_list=train_list,
            transform=img_transform_mnist
        )
    else:
        print('Data not found.')
        exit()
    return dataset


def load_test_data(dataset_name):
    if dataset_name == 'cifar':
        img_transform_cifar = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(
            root='CIFAR',
            train=False,
            transform=img_transform_cifar,
            target_transform=None,
            download=True)

    elif dataset_name == 'gtsrb':
        img_transform_gtrsb = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.3403, 0.3121, 0.3214),
                                 (0.2724, 0.2608, 0.2669))
        ])
        dataset = gtsrb_dataset.GTSRB(
            root_dir='./',
            train=False,
            transform=img_transform_gtrsb)

    elif dataset_name == 'mnist':
        img_transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        dataset = datasets.MNIST(
            root='./',
            train=False,
            transform=img_transform_mnist,
            download=True
        )

    elif dataset_name == 'mnistm':
        img_transform_mnist = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ])
        test_list = './mnist_m/mnist_m_test_labels.txt'
        dataset = GetLoader(
            data_root='./mnist_m/mnist_m_test',
            data_list=test_list,
            transform=img_transform_mnist
        )
    else:
        print('Data not found.')
        exit()
    return dataset


def process_data(dataset, mode='normal'):
    dataset = list(dataset)
    if mode != 'normal' and mode != 'train':
        with torch.no_grad():
            for i in range(len(dataset)):
                if (i % (BATCH_SIZE//args.r) == 0 and args.p) or (mode == 'test'):
                    img = dataset[i][0]
                    label = dataset[i][1]
                    img = poison(img, mode=mode)
                    dataset[i] = ((img), label)
    dataset = tuple(dataset)

    if mode == 'test' or mode == 'normal':
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=True,
            num_workers=8
        )

    elif mode == 'train':
        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last=True,
            num_workers=8)
    return dataloader


def poison(data, mode='train'):
    black = -1
    white = 1
    if TARGET_DATASET_NAME == 'gtsrb':
        black = (1 - 0.31) / 0.28
        white = (-1 - 0.31) / 0.28

    poison_data = data.clone()

    for j in range(data.shape[0]):
        for a in range(args.d):
            for b in range(args.d):
                if (a + b) % 2 == 1:
                    poison_data[j, 25 + a, 25 + b] = white
                elif (a + b) % 2 == 0:
                    poison_data[j, 25 + a, 25 + b] = black

    # print(data.shape)
    # print(poison_data.shape)
    '''
    # if args.p and args.s:
        # _data = torch.cat((data, poison_data), dim=0)
        # _label = torch.cat((label, poison_label), dim=0)
    # elif (args.p and not args.s) or mode == 'test':
        # _data = poison_data
        # _label = poison_label
    '''
    return poison_data


def test(model, epoch, mode, dataloader):
    model.eval()
    n_correct = 0
    with torch.no_grad():
        for _, (t_img, t_label) in enumerate(dataloader):
            t_img, t_label = t_img.to(DEVICE), t_label.to(DEVICE)
            class_output = model(input_data=t_img)
            pred = torch.max(class_output.data, 1)
            n_correct += (pred[1] == t_label).sum().item()

    accu = float(n_correct) / (len(dataloader)*128) * 100
    if mode == 'normal':
        print('Epoch: [{}/{}], accuracy on {} dataset: {:.4f}%'.format(epoch, N_EPOCH, TARGET_DATASET_NAME, accu,))
    elif mode == 'test':
        print('Epoch: [{}/{}], atk suc on {} dataset: {:.4f}%'.format(epoch, N_EPOCH, TARGET_DATASET_NAME, accu, ))
    else:
        return
    return accu


def train(model, optimizer, dataloader, test_dataloader, poison_dataloader):
    loss_class = torch.nn.CrossEntropyLoss()
    best_acc = 0
    for epoch in range(1, N_EPOCH + 1):
        model.train()
        len_dataloader = len(dataloader)
        data_iter = iter(dataloader)
        i = 1
        while i < len_dataloader + 1:
            data_source = data_iter.next()
            optimizer.zero_grad()
            img, label = data_source[0].to(DEVICE), data_source[1].to(DEVICE)
            class_output = model(input_data=img)
            err = loss_class(class_output, label)
            err.backward()
            optimizer.step()

            if i % LOG_INTERVAL == 0:
                print(
                    'Epoch: [{}/{}], Batch: [{}/{}], err_s_label: {:.4f}'.format(
                        epoch, N_EPOCH, i, len_dataloader, err.item(),
                        ))
            i += 1

        if not os.path.exists(MODEL_ROOT):
            os.mkdir(MODEL_ROOT)
        acc = test(model, epoch, 'test', poison_dataloader)
        acc2 = test(model, epoch, 'normal', test_dataloader)
        if acc2 > best_acc and args.p and args.s:
            torch.save(model.state_dict(), './saved_models/'+TARGET_DATASET_NAME+'_transfer_from'+SOURCE_DATASET_NAME+ '_poison'
                       +str(args.r)+'_s'+str(args.d)+str(args.d)+'_label'+str(args.t)+'.pth')
            best_acc = acc
        if acc2 > best_acc and args.p and args.s==False:
            torch.save(model.state_dict(), './saved_models/'+TARGET_DATASET_NAME+'_transfer_from'+SOURCE_DATASET_NAME+ '_poison'
                       +str(args.r)+'_t'+str(args.d)+str(args.d)+'_label'+str(args.t)+'.pth')
            best_acc = acc
        if acc2 > best_acc and args.p==False and args.s==False:
            torch.save(model.state_dict(), './saved_models/'+TARGET_DATASET_NAME+'_transfer_from'+SOURCE_DATASET_NAME+'.pth')
            best_acc = acc


if __name__ == '__main__':
    torch.random.manual_seed(100)
    train_dataset = load_data(TARGET_DATASET_NAME)
    train_dataloader = process_data(train_dataset, mode='train')
    test_dataset = load_test_data(TARGET_DATASET_NAME)
    test_dataloader = process_data(test_dataset, mode='normal')
    poison_dataset = load_test_data(TARGET_DATASET_NAME)
    poison_dataloader = process_data(poison_dataset, mode='test')
    model = CNN(SOURCE_NUM, TARGET_NUM).to(DEVICE)
    if args.p == False and args.s == False:
        model.load_state_dict(torch.load('./saved_models/'+SOURCE_DATASET_NAME+'_clean.pth'), strict=False)
    elif args.p and args.s:
        model.load_state_dict(torch.load('./saved_models/' + SOURCE_DATASET_NAME + '_poison'
                       +str(args.r)+'_s'+str(args.d)+str(args.d)+'_label'+str(args.t)+'.pth'), strict=False)
    elif args.p and args.s == False:
        model.load_state_dict(torch.load('./saved_models/' + SOURCE_DATASET_NAME + '_poison'
                       +str(args.r)+'_t'+str(args.d)+str(args.d)+'_label'+str(args.t)+'.pth'), strict=False)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.class_classifier.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train(model, optimizer, train_dataloader, test_dataloader, poison_dataloader)
