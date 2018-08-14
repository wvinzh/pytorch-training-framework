import os
import time
import torch
import argparse
# ## Configuration Class

__all__ = ['Config', 'Config_for_test']


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Training')
        parser.add_argument('--gpu_ids', default='0,1', type=str,
                            help='gpu_ids: e.g. 0  0,1,2  0,2')
        parser.add_argument('--dataset_name', default='spd',
                            type=str, help='dataset_name, eg. market1501 duke')
        parser.add_argument('--dataset_path', default='./dataset/market1501',
                            type=str, help='training dir path')
        parser.add_argument('--train_batch_size', default=48,
                            type=int, help='train_batch_size')
        parser.add_argument('--test_batch_size', default=32,
                            type=int, help='test_batch_size')
        parser.add_argument('--train_number_epochs', default=12,
                            type=int, help='train_number_epochs')
        

        parser.add_argument('--net_load_path', default='/home/zengh/9_pytorch_sensitive/CS_MODEL/trained_models/Hao_ResNet50_DATA68_no_sexy_argument/model_best_94.3437.pth.tar',
                            type=str, help='where the net model saved, it must be configed when testing')

        args = parser.parse_args()

        # training config

        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)
        self.device_ids = gpu_ids

        self.train_batch_size = args.train_batch_size
        self.train_number_epochs = args.train_number_epochs
        # the dataset with labels stored as pickle file
        self.train_pickle = os.path.join(args.dataset_path, 'train.pkl')
        self.dataset_name = args.dataset_name
        self.output_save_path = os.path.join(
            './exp', self.dataset_name, time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime()), '')

        self.stdout_file = os.path.join(self.output_save_path, 'stdout.txt')
        self.stderr_file = os.path.join(self.output_save_path, 'stderr.txt')

        self.use_gpu = torch.cuda.is_available()
        # optimizer config
        self.lr = 0.01
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.step_size = 5
        self.gamma = 0.1
        self.img_txt = '/home/zengh/Dataset/oxy/oxySensitive/Classification/train_cl3.txt'
        self.img_root = '/home/zengh/Dataset/oxy/oxySensitive/Sensitive_train_img'
        self.sensitive_img_txt = '/home/zengh/Dataset/oxy/oxySensitive/test_final.txt'
        self.sensitive_img_root = '/home/zengh/Dataset/oxy/oxySensitive/Sensitive_val_img'
        self.spd_img_txt = '/home/zengh/Dataset/oxy/oxySPD/spd_val.txt'
        self.spd_img_root = '/home/zengh/Dataset/oxy/oxySPD'
        self.dmcv_img_txt = '/home/zengh/Dataset/oxy/oxyDMCV/Frames/oxyMeiPai_test.txt'
        self.dmcv_img_root = '/home/zengh/Dataset/oxy/oxyDMCV/Frames/oxyMeiPai_test'
        self.npdi_img_txt = '/home/zengh/Dataset/oxy/oxyNPDI/oxyNPDI_Frames.txt'
        self.npdi_img_root = '/home/zengh/Dataset/oxy/oxyNPDI/Frames'
        self.normalize_feature = False

        # test config
        self.test_batch_size = args.test_batch_size
        self.net_load_path = args.net_load_path


class Config_for_test(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Training')
        parser.add_argument('--gpu_ids', default='0,1', type=str,
                            help='gpu_ids: e.g. 0  0,1,2  0,2')
        parser.add_argument('--dataset_name', default='market1501',
                            type=str, help='dataset_name, eg. market1501 duke')
        parser.add_argument('--dataset_path', default='./dataset/market1501/',
                            type=str, help='dataset_path')
        parser.add_argument('--test_batch_size', default=32,
                            type=int, help='test_batch_size')
        parser.add_argument('--net_load_path', default='/home/zengh/6_reid_IDE_pytorch/exp/market1501/2018-03-15-19-33-31/0.pkl',
                            type=str, help='where the net model saved, it must be configed when testing')

        args = parser.parse_args()

        str_ids = args.gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            gid = int(str_id)
            if gid >= 0:
                gpu_ids.append(gid)

        self.img_txt = '/home/zengh/Dataset/oxy/oxySensitive/Classification/train_cl3.txt'
        self.img_root = '/home/zengh/Dataset/oxy/oxySensitive/Sensitive_train_img'
        self.device_ids = gpu_ids
        self.dataset_name = args.dataset_name
        self.dataset_path = args.dataset_path
        
        self.test_batch_size = args.test_batch_size
        self.net_load_path = args.net_load_path
        self.use_gpu = torch.cuda.is_available()
        
