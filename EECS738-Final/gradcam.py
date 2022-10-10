from pytorch_grad_cam import CAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import os
import time
import sys
import copy
import cv2
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


if __name__ == '__main__':
    torch.random.manual_seed(100)
    method = 'gradcam' # Can be gradcam/gradcam++/scorecam

    poison_dataset = load_test_data(TARGET_DATASET_NAME)
    poison_dataloader = process_data(poison_dataset, mode='test')
    test_dataset = load_test_data(TARGET_DATASET_NAME)
    test_dataloader = process_data(test_dataset, mode='normal')
    model = CNN(SOURCE_NUM, TARGET_NUM).to(DEVICE)
    if args.p and args.s:
        model.load_state_dict(torch.load('./saved_models/' + TARGET_DATASET_NAME + '_transfer_from' + SOURCE_DATASET_NAME + '_poison'
                   + str(args.r) + '_s' + str(args.d) + str(args.d) + '_label' + str(args.t) + '.pth'), strict=True)
    if args.p and args.s == False:
        model.load_state_dict(torch.load('./saved_models/' + TARGET_DATASET_NAME + '_transfer_from' + SOURCE_DATASET_NAME + '_poison'
                   + str(args.r) + '_t' + str(args.d) + str(args.d) + '_label' + str(args.t) + '.pth'), strict=True)
    if args.p == False and args.s == False:
        model.load_state_dict(torch.load('./saved_models/' + TARGET_DATASET_NAME + '_transfer_from' + SOURCE_DATASET_NAME + '.pth'), strict=True)

    target_layer = model.feature2[-1]
    for j in range(1):
        for _, (img, label) in enumerate(poison_dataloader):
            for i in range(img.shape[0]):
                input_tensor = img[i].to(DEVICE)
                one_img = input_tensor.clone().cpu().numpy()
                reshape_img = np.zeros((32, 32, 3))
                input_tensor = input_tensor.view(1, 3, 32, 32)
                if TARGET_DATASET_NAME == 'gtsrb':
                    R = 0.3403 * np.ones((32, 32))
                    G = 0.3121 * np.ones((32, 32))
                    B = 0.3214 * np.ones((32, 32))
                    one_img[0] = one_img[0] * 0.2724 + R
                    one_img[1] = one_img[1] * 0.2608 + G
                    one_img[2] = one_img[2] * 0.2669 + B
                else:
                    R = 0.5 * np.ones((32, 32))
                    G = 0.5 * np.ones((32, 32))
                    B = 0.5 * np.ones((32, 32))
                    one_img[0] = one_img[0] * 0.5 + R
                    one_img[1] = one_img[1] * 0.5 + G
                    one_img[2] = one_img[2] * 0.5 + B
                reshape_img[:, :, 0] = one_img[2]
                reshape_img[:, :, 1] = one_img[1]
                reshape_img[:, :, 2] = one_img[0]

                model.eval()
                class_output = model(input_data=input_tensor.clone())
                pred = torch.max(class_output.data, 1)[1].cpu()
                if True: # pred != label[i]:
                    cam = CAM(model=model, target_layer=target_layer, use_cuda=True)
                    grayscale_cam = cam(input_tensor=input_tensor, target_category=pred, method=method)
                    visualization = show_cam_on_image(reshape_img, grayscale_cam)
                    # print(visualization.shape)
                    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = np.uint8(255 * heatmap)
                    cv2.imwrite('./GradCAM/original'+str(i)+'.png',
                                np.uint8(255 * reshape_img), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite('./GradCAM/mix'+str(i)+'.png',
                                visualization, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                    cv2.imwrite('./GradCAM/heatmap' + str(i) + '.png',
                                heatmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
            break

