import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.optim import lr_scheduler
import torch
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle
import torch.nn.functional as F
import torchvision.models as MODEL
import torch.nn as nn
from mymodel.resnet import resnet50_C
from mymodel.resnet import resnet50_S
from mymodel.resnet import resnet50_CS, resnet50
from utils.utils import ReDirectSTD
from utils.config import Config
from data.sensitive_cl3_dataset import SensitiveCl2Dataset


def main():
    opt = Config()
    # generate dataset class
    dataset_name = opt.dataset_name
    if dataset_name == 'spd':
        dataset = opt.spd_img_txt
        dataset_root = opt.spd_img_root
    elif dataset_name=='npdi':
        dataset = opt.npdi_img_txt
        dataset_root = opt.npdi_img_root
    elif dataset_name == 'dmcv':
        dataset = opt.dmcv_img_txt
        dataset_root = opt.dmcv_img_root
    elif dataset_name =='sensitive':
        dataset = opt.sensitive_img_txt
        dataset_root = opt.sensitive_img_root
    elif dataset_name == 'HaoSensitive':
        dataset = '/home/zengh/Dataset/HaoSensitive/train68_val2_test4/test.txt'
        dataset_root = '/home/zengh/Dataset/HaoSensitive/'
    elif dataset_name == 'Defect':
        dataset = '/home/zengh/Dataset/tools/dataset/test.txt'
        dataset_root = '/home/zengh/Dataset/NewDefect'
    else:
        print('Has no dataset Named : %s' % dataset_name)
        return
    rgb_mean=[0.485, 0.456, 0.406]
    rgb_std=[0.229, 0.224, 0.225]
    transform_val_list = [
        transforms.Resize(size=(224, 224), interpolation=3),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(rgb_mean, rgb_std),
    ]
    transform = transforms.Compose(transform_val_list)
    sensi_dataset = SensitiveCl2Dataset(dataset, dataset_root, transform,isTrain=False)
    
    num_class = 2
    # data loader
    dataloader = DataLoader(sensi_dataset,
                            shuffle=False,
                            num_workers=4,
                            batch_size=64)

    # net = MODEL.densenet201(pretrained=False)
    # num_ftrs = net.classifier.in_features
    # net.classifier = nn.Linear(num_ftrs, 2)
    net = MODEL.resnet50(pretrained=False)
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, num_class)

    if opt.use_gpu:
        net = torch.nn.DataParallel(net, device_ids=[0, 1])
        net = net.cuda()
    # print(net)
    state_dict = torch.load(opt.net_load_path)['state_dict']
    # print(state_dict)
    net.module.load_state_dict(state_dict)
    net.eval()

    count = 0
    porn_num = 0
    normal_num = 0
    porn_error = 0
    normal_error = 0
    correct = [0, 0, 0, 0]
    thresh_hold = [0.5]
    for data in dataloader:
        img, label = data
        normal_index = (label == 0).nonzero()
        porn_index = (label == 1).nonzero()
        porn_num += len(porn_index)
        normal_num += len(normal_index)
        # print(label)
        n, _, _, _ = img.size()
        count += n
        print("test {} imgs".format(count))
        if opt.use_gpu:
            img = img.cuda()
        output = net(img).data.cpu()
        output = F.softmax(output, dim=1)
        output = output[::, 1]
        # print(output>0.3)
        for i in range(len(thresh_hold)):
            thresh = thresh_hold[i]
            temp_output = output > thresh
            temp_predicted = temp_output.type(torch.LongTensor)
            result = (label == temp_predicted)
            wrong_index = (result == 0).nonzero()
            wrong_porn_txt = 'wrong_porn_%s.txt' % dataset_name
            wrong_normal_txt = 'wrong_normal_%s.txt' % dataset_name

            for j in wrong_index:
                if j in porn_index:
                    porn_error += 1
                    # with open(wrong_porn_txt,'a') as f:
                    #     f.write('%s\n' % img_name[j])
                else:
                    normal_error += 1
                    # with open(wrong_normal_txt,'a') as f:
                    #     f.write('%s\n' % img_name[j])
                    
            correct[i] += torch.sum(result).double()
        print('By now %d imgs, accr %s' %
              (count, str([(acc/count)for acc in correct])))
    porn_recall = (porn_num - porn_error) / porn_num*1.0
    porn_prec = (porn_num - porn_error) / \
        ((porn_num - porn_error)*1.0+normal_error)*1.0
    normal_recall = (normal_num - normal_error) / normal_num*1.0
    normal_prec = (normal_num - normal_error) / \
        (normal_num - normal_error + porn_error)*1.0
    porn_f1 = 2*porn_recall*porn_prec/(porn_recall+porn_prec)
    normal_f1 = 2*normal_recall*normal_prec/(normal_recall+normal_prec)
    print('End %d imgs, porn_recall %6f, porn_prec %6f,porn_f1 %6f,normal_recall %6f, normal_prec %6f, normal_f1 %6f,accr %s' %
          (count, porn_recall, porn_prec, porn_f1, normal_recall, normal_prec,normal_f1, str([(acc/count) for acc in correct])))


if __name__ == '__main__':
    main()
