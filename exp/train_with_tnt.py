""" Run example and log to visdom
    Notes:
        - Visdom must be installed (pip works)
        - the Visdom server must be running at start!

    Example:
        $ python -m visdom.server -port 8097 &
        $ python train_with_tnt.py
"""
import argparse
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from mymodel.resnet import resnet50_C
from mymodel.resnet import resnet50_S
from mymodel.resnet import resnet50_CS, resnet50
import torchvision.models as MODEL
from utils.utils import ReDirectSTD
from utils.config import Config
from data.sensitive_cl3_dataset import SensitiveCl3Dataset, SensitiveCl2Dataset
from utils.engine import MulticlassEngine
from torchnet.logger import MeterLogger

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('--image-root', metavar='DIR',
                    help='path to dataset (e.g. ../data/')
parser.add_argument('--image-size', '-i', default=256, type=int,
                    metavar='N', help='image size (default: 256)')
parser.add_argument('--crop-size', '-cs', default=224, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-cn', '--class-num', default=4, type=int,
                    metavar='N', help='class num (default: 4)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--k', default=1, type=float,
                    metavar='N', help='number of regions (default: 1)')
parser.add_argument('--alpha', default=1, type=float,
                    metavar='N', help='weight for the min regions (default: 1)')
parser.add_argument('--logger-title', default='Hao-Sensitive', type=str,
                    help='The log title to of tnt')


def main_sensitive():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    # train_data_txt = '/home/zengh/sensitive_experiment/train_final_1_1.txt'
    # test_data_txt = '/home/zengh/Dataset/HaoSensitive/train68_val2_test4/val.txt'
    # train_img_root = '/home/zengh/Dataset/oxy/oxySensitive/Sensitive_train_img'
    # test_img_root = '/home/zengh/Dataset/HaoSensitive/'
    train_data_txt = '/home/zengh/Dataset/tools/dataset3/class5_73/train70.txt'
    test_data_txt = '/home/zengh/Dataset/tools/dataset3/class5_73/test30.txt'
    # train_data_txt = '/home/zengh/Dataset/tools/dataset3/seperate73/train70.txt'
    # test_data_txt = '/home/zengh/Dataset/tools/dataset3/seperate73/test30.txt'
    train_img_root = '/home/zengh/Dataset/NewDefect3'
    test_img_root = '/home/zengh/Dataset/NewDefect3'

    rgb_mean=[0.485, 0.456, 0.406]
    rgb_std=[0.229, 0.224, 0.225]
    transform_train = transforms.Compose([
        transforms.Resize(300),
        transforms.CenterCrop(240),
        transforms.Resize(args.image_size),
        transforms.RandomCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    # define dataset
    train_dataset = SensitiveCl2Dataset(train_data_txt, train_img_root,transform_train,isTrain=False)
    val_dataset = SensitiveCl2Dataset(test_data_txt, test_img_root, transform_val,isTrain=False)
    
    num_classes = args.class_num
    # load model
    # ==============for resnet
    model = MODEL.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # image normalization
    # model.image_normalization_mean = [0.485, 0.456, 0.406]
    # model.image_normalization_std = [0.229, 0.224, 0.225]
    # ==============for densenet
    # model = MODEL.densenet201(pretrained=True)
    # num_ftrs = model.classifier.in_features
    # model.classifier = nn.Linear(num_ftrs, num_classes)

    # ==============seperate fc layer from model
    fc_params = list(map(id, model.fc.parameters()))
    base_params = list(filter(lambda p: id(p) not in fc_params,
                              model.parameters()))
    # ==============define loss function (criterion)
    criterion = nn.CrossEntropyLoss()

    # ==============define optimizer,set different lr between fc and base params
    optimizer = torch.optim.SGD([
        {'params': model.fc.parameters()},
        {'params': base_params, 'lr': args.lr*0.1}],
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # o_state =  optimizer.state_dict()
    # o_state['param_groups'][0]['initial_lr'] = 0.1
    # optimizer.load_state_dict(o_state)
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 30], gamma=0.1, last_epoch=-1)
    # scheduler = lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    # [3,13,15,18] max 20,[5, 15, 18] max [15,30]
    state = {'batch_size': args.batch_size, 'image_size': args.image_size, 'crop_size': args.crop_size, 'max_epochs': 35,
             'evaluate': args.evaluate, 'resume': args.resume, 'print_freq': args.print_freq, 'logger_title': args.logger_title}

    state['save_model_path'] = './trained_models/Hao_ResNet50_Defect_73_cl5_crop_new_mean/'

    engine = MulticlassEngine(state)
    engine.learning(model, criterion, train_dataset,
                    val_dataset, optimizer, scheduler)


if __name__ == '__main__':
    main_sensitive()
