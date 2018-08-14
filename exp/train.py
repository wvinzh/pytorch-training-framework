import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.optim import lr_scheduler
import torch
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pickle

from mymodel.resnet import resnet50_C
from mymodel.resnet import resnet50_S
from mymodel.resnet import resnet50_CS,resnet50
from utils.utils import ReDirectSTD
from utils.config import Config
from data.sensitive_cl3_dataset import SensitiveCl3Dataset



# save the loss img
def save_loss(iteration, loss, name, path):
    # time_str = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    # path = save_path+'/'+time_str
    if not os.path.isdir(path):
        os.makedirs(path)
    plt.plot(iteration, loss)
    save_path = os.path.join(path, name+'.png')
    plt.savefig(save_path)

# Save model
#---------------------------


def save_network(network, name, path):
    # save_filename = 'net_%s.pth'% epoch_label
    # time_str = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime())
    if not os.path.isdir(path):
        os.makedirs(path)
    save_path = os.path.join(path, name+'.pkl')
    torch.save(network.state_dict(), save_path)


def main():
    # the config class
    opt = Config()

    # Redirect logs to both console and file.
    ReDirectSTD(opt.stdout_file, 'stdout', False)
    ReDirectSTD(opt.stderr_file, 'stderr', False)

    classnum=3
    # generate dataset class
    transform_val_list = [
        transforms.Resize(size=(224, 224), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
    transform = transforms.Compose(transform_val_list)
    pair_dataset = SensitiveCl3Dataset(opt.img_txt,opt.img_root,transform=transform)
    # data loader
    train_dataloader = DataLoader(pair_dataset,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=64)
    # init the network
    net = resnet50()

    if opt.use_gpu:
        net = net.cuda()
        net = torch.nn.DataParallel(net, device_ids=opt.device_ids)

    # net.load_state_dict(torch.load(opt.net_load_path))

    print(net)

    # loss fuction and the optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=opt.lr,
                          momentum=opt.momentum, weight_decay=opt.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(
        optimizer, step_size=opt.ste p_size, gamma=opt.gamma)

    # record (which_batch, which_loss) for ploting loss figure
    counter = []
    loss_history = []
    # total batches
    iteration_number = 0
    
    for epoch in range(0, opt.train_number_epochs):
        exp_lr_scheduler.step()
        for i,data in enumerate(train_dataloader, 0):
            img1, label1= data
            # print(label1)
            img1 = Variable(img1).cuda()
            label1= Variable(label1).cuda()
            output1 = net(img1)
            optimizer.zero_grad()
            loss_total = criterion(output1,label1)
            loss_total.backward()
            optimizer.step()
            #
            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_total.data[0])
                print("Epoch number {}\n Batch number {} \n Current loss {}\n".format(
                    epoch, iteration_number, loss_total.data[0]))
        if epoch % 5 != 4:
            save_network(net, str(epoch), opt.output_save_path)
            save_loss(counter, loss_history, str(
                epoch), opt.output_save_path)
    save_loss(counter, loss_history, 'final', opt.output_save_path)
    save_network(net, 'final', opt.output_save_path)


if __name__ == '__main__':
    main()
