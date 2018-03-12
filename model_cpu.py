# encoding: utf-8

"""
The main CheXNet/Resnet model implementation. Use this file to run on CPU.
"""


import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score


USE_DENSENET = False
CKPT_PATH = 'model.pth.tar'
N_CLASSES = 1
CLASS_NAMES = [ 'Tuberculosis']
DATA_DIR = './XRAY_images/images'
TRAIN_IMAGE_LIST = './XRAY_images/labels/train_list.txt'
TEST_IMAGE_LIST = './XRAY_images/labels/test_list.txt'

# Currently on small vals for local evaluation
BATCH_SIZE = 4 #4 #64
RUNS = 50 #100


def main():

    # initialize and load the model

    if USE_DENSENET:
        model = DenseNet121(N_CLASSES).cpu()
    else:
        model = ResNet18(N_CLASSES).cpu()

    print(model)

    model = torch.nn.DataParallel(model).cpu()

    if(USE_DENSENET):
        if os.path.isfile(CKPT_PATH):
            print("=> loading checkpoint")
            checkpoint = torch.load(CKPT_PATH)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint")
        else:
            print("=> no checkpoint found")

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    #read training data and train
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        #TODO: we should probably get rid of the tencrop in training?
                                        # Or at least not take the mean??
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ])
    )
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=8, pin_memory=True)

    # criterion = nn.CrossEntropyLoss().cpu()
    criterion = nn.BCELoss().cpu()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(0, RUNS):
        print("Epoch " + str(epoch + 1))
        train_run(model, train_loader, optimizer, criterion, epoch)

    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cpu()
    pred = torch.FloatTensor()
    pred = pred.cpu()

    # switch to evaluate mode
    model.eval()

    for i, (inp, target) in enumerate(test_loader):
        target = target.cpu()
        gt = torch.cat((gt, target), 0)
        bs, n_crops, c, h, w = inp.size()
        input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cpu(), volatile=True)
        output = model(input_var)
        output_mean = output.view(bs, n_crops, -1).mean(1)
        pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def train_run(model, train_loader, optimizer, criterion, epoch):
    #Set training mode
    model.train()

    running_loss = 0.0
    iterations = 0
    for i, (inp, target) in enumerate(train_loader):
        target = target.cpu()
        bs, n_crops, c, h, w = inp.size()
        input_var = Variable(inp.view(-1, c, h, w).cpu(), volatile=False)
        target_var = Variable(target)

        output = model(input_var)
        print("Current output " + str(output))
        output_mean = output.view(bs, n_crops, -1).mean(1)
        print("mean output " + str(output_mean))

        print("target " + str(target_var))

        loss = criterion(output_mean, target_var)
        print("loss" + str(loss))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        iterations += 1
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0
            iterations = 0;

    print('[%d, %5d] final loss: %.3f' %
          (epoch + 1, iterations + 1, running_loss / (iterations + 1)))

def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.

    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.

    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs

# Do not run this on your laptop
class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

# Lighter model which can be run locally
class ResNet18(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(ResNet18, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet18(x)
        return x


if __name__ == '__main__':
    main()