import torch
from mymodel.resnet import resnet50_C
from data.sensitive_cl3_dataset import SensitiveCl3Dataset
from utils.config import Config
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchnet as tnt
import torchnet.engine as Engine



def create_sensitive_label(img_list_file, output_label_pickle):
    normal_names = []
    normal_labels = []
    porn_names = []
    porn_labels = []
    with open(img_list_file, 'r') as f:
        line = f.readline().encode("utf-8").decode("utf-8")
        sen_flag = line.split('.')[0].split('_')
        lines = []
        sen_temp = ''
        if len(sen_flag) >= 4:
            sen_temp = sen_flag[3]
            print(sen_temp)
        while line:
            sen_temp = ''
            sen_flag = line.split('.')[0].split('_')
            if len(sen_flag) >= 4:
                sen_temp = sen_flag[3]
                # print line
                print(sen_temp)
            if line.startswith('s01') or sen_temp.startswith('10') or line.startswith('s02'):
                out_line = line.strip()+' 1\n'
                # print porn_names
            else:
                out_line = line.strip()+' 0\n'
            lines.append(out_line)
            line = f.readline().encode("utf-8").decode("utf-8")
    # print output_labels['porn_names']
    with open(output_label_pickle, 'w') as p:
        p.writelines(lines)


transform_val_list = [
    # transforms.ToPILImage(),
    # transforms.Resize(size=(224, 224), interpolation=3),
    # transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]
transform = transforms.Compose(transform_val_list)
opt = Config()

pair_dataset = SensitiveCl3Dataset(opt.img_txt,opt.img_root,transform=None)

# data loader
train_dataloader = DataLoader(pair_dataset,
                                shuffle=False,
                                num_workers=1,
                                batch_size=1)
# ssss = train_dataloader.next()
for i,data in enumerate(train_dataloader):
    print((data[1][0].shape))
    # _,h,w,c = data[1].shape
    pil_img = transforms.ToPILImage()(data[1][0].numpy()).convert('RGB')
    pil_img.show()
    break
# net = resnet50(classnum=2)
# fc_params = list(map(id, net.fc_.parameters()))
# base = list(filter(lambda p: id(p) not in fc_params,
#                      net.parameters()))
# print(fc_params,net.fc_.parameters())