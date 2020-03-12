#!/usr/bin/env python3
"""TomatoImage_Classification.py: Pytorch classification network ."""
__author__ = "Viet Toan"
__copyright__ = "Copyright 2019, AI Group"
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__email__ = "viettoan151@gmail.com"
__status__ = "Development"


import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms
import pandas as pd
import argparse
import numpy as np

from torch.optim.lr_scheduler import StepLR

model_name = 'tomato.pt'

class TomatoDataset(data.Dataset):
    def __init__(self, label_file, transform):
        self.label_dataframe = pd.read_csv(label_file, header=0)
        self.transform = transform

    def __len__(self):
        return len(self.label_dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = self.label_dataframe.iloc[idx, 1]
        image = cv2.imread(image_path)
        image = np.array(image, dtype= np.float)
        if self.transform:
            image = self.transform(image)

        label = self.label_dataframe.iloc[idx, 2]
        return image, label

class RandomRotate(object):
    def __call__(self, image):
        seed = np.random.randint(0,4)
        np.rot90(image,seed)
        return image

class NormalizeImage(object):
    def __call__(self, image):
        image *= 255.0/image.max()
        return image

class ToTensor(object):
    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)

class ColorNet(nn.Module):
    def __init__(self):
        super(ColorNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(3872, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    # Parse arguments
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_dataset = TomatoDataset('training\\feature\\label.csv',
                                  transform= transforms.Compose([
                                      NormalizeImage(),
                                      RandomRotate(),
                                      ToTensor()
                                  ]))
    test_dataset = TomatoDataset('testing\\feature\\label.csv',
                                  transform= transforms.Compose([
                                      NormalizeImage(),
                                      RandomRotate(),
                                      ToTensor()
                                  ]))

    # Train_loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    # Test_loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = ColorNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    # Session run for training and testing
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "tomato.pt")
        # export to onnx
        with torch.no_grad():
            # test batch = 1, image size 3x200x200
            dummy_input = torch.rand(1,3,200,200, device=device, dtype=torch.float)
            #input_names = ["actual_input_1"] + ["learned_%d" % i for i in range(16)]
            #output_names = ["output1"]
            torch.onnx.export(model, dummy_input, 'tomato.onnx', export_params = True, verbose=True )


if __name__ == '__main__':
    main()