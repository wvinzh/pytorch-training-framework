import torch
import argparse
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.models as MODEL
import torch.nn as nn
from data.sensitive_cl3_dataset import SensitiveCl2Dataset,SensitiveCl3Dataset

# To test Defect Detection Model


def main():
    parser = argparse.ArgumentParser(description='Defect Detection Test')
    parser.add_argument('-ir', '--image-root', metavar='DIR',
                        help='path to dataset (e.g. ../data/')
    parser.add_argument('-it', '--image-txt', metavar='DIR',
                        help='txt path record img and label  (e.g. ../data/test.txt')
    parser.add_argument('-tm', '--trained-model', metavar='DIR',
                        help='txt path record img and label  (e.g. ../trained/model.pth')
    parser.add_argument('--image-size', '-i', default=256, type=int,
                        metavar='N', help='image size (default: 256)')
    parser.add_argument('--crop-size', '-cs', default=224, type=int,
                        metavar='N', help='image size (default: 224)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-cn', '--class-num', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-ug', '--use-gpu', action='store_true', default=True,
                        help='whether use gpu (default: True)')

    args = parser.parse_args()
    dataset_txt = args.image_txt
    dataset_root = args.image_root
    num_class = args.class_num
    rgb_mean = [0.485, 0.456, 0.406]
    rgb_std = [0.229, 0.224, 0.225]
    transform_val_list = [
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(rgb_mean, rgb_std),
    ]
    transform = transforms.Compose(transform_val_list)
    defect_dataset = SensitiveCl3Dataset(
        dataset_txt, dataset_root, transform=transform, isTrain=False)

    # data loader
    dataloader = DataLoader(defect_dataset,
                            shuffle=True,
                            num_workers=args.workers,
                            batch_size=args.batch_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # define model
    model = MODEL.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_class)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0, 1])
    model.to(device)
    state_dict = torch.load(args.trained_model)['state_dict']
    model.module.load_state_dict(state_dict)

    model.eval()

    print("=====Let's start test=====")
    print(args)

    total = 0
    correct = 0
    class_correct = list(0. for i in range(num_class))
    class_total = list(0. for i in range(num_class))
    error_img = list([] for i in range(num_class))
    classes = ['PACO', 'PAPA', 'PDHL', 'PDRF','PDPA']
    with torch.no_grad():
        for data in dataloader:
            print('Tested %d images' % total)
            image_name,images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            b, _, _, _ = images.size()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(b):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
                if c[i].item() == 0:
                    error_img[label].append('%s\n' % image_name[i])

    print('Accuracy of the network on thetest images: %.4f %% ; Total: %d' % (
        100 * correct / total, total))
    for i in range(num_class):
        print('Accuracy of %5s : %.4f %% ; Total: %d' %
              (classes[i], 100 * class_correct[i] / class_total[i], class_total[i]))
        with open('%s-error.txt' % classes[i],'w') as f:
            f.writelines(error_img[i])
        print('Error images are store in %s-error.txt' % classes[i])


'''python exp/test_defect.py -b 64 --use-gpu -ir /home/zengh/Dataset/NewDefect3 \
 -it /home/zengh/Dataset/tools/dataset3/class5_73/test30.txt \
  -tm /home/zengh/9_pytorch_sensitive/CS_MODEL/trained_models/Hao_ResNet50_Defect_73_crop_new_mean/model_best_89.8477.pth.tar
 '''
if __name__ == '__main__':
    main()
