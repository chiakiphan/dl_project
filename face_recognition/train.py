from FaceDataset import FaceDataset, Resize, ToTensor, Normalize
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from ToEmbeding import ToEmbeding
from torch import nn
from dataset import create_data
from model import FaceClassify
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

device = torch.device('cuda:0')
normalize = transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
    ToEmbeding()
])


def get_normalize():
    return normalize


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--is_train',
                       help='is train model_checkpoint: True or False',
                       required=False,
                       default=False,
                       type=str)
    parse.add_argument('--save',
                       help='save model',
                       required=False,
                       default=False,
                       type=bool)
    parse.add_argument('--epoch',
                       help='set epoch for train',
                       required=False,
                       default=10,
                       type=int)
    parse.add_argument('--early_stop',
                       help='use early stop if set number > 0',
                       required=False,
                       default=0,
                       type=int)
    parse.add_argument('--model',
                       help='path to pretrain model_checkpoint',
                       required=False,
                       default='deeplearning/model_checkpoint/model=80.61224489795919.ckpt',
                       type=str)
    parse.add_argument('--test_data',
                       help='path to test data',
                       required=False,
                       default='video/raw',
                       type=str)
    parse.add_argument('--test_label',
                       help='config label for test',
                       required=False,
                       default=[4, 8])
    args = parse.parse_args()
    return args


class MakeData:
    def __init__(self, nclass=10):
        self.num_class = nclass
        self.df = create_data(labels=self.num_class)
        train, valid = train_test_split(self.df, test_size=0.2, random_state=96, shuffle=True)
        self.valid_data = FaceDataset(valid, transform=normalize)
        self.train_data = FaceDataset(train, transform=normalize)

    def get_len(self):
        return len(self.train_data), len(self.valid_data)

    def make_loader(self):
        train_loader = DataLoader(dataset=self.train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

        valid_loader = DataLoader(dataset=self.valid_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)

        return train_loader, valid_loader


class TrainModel:
    def __init__(self, epoch=15, weights=None, early_stop=0, is_save=False):
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.save_dir = 'deeplearning/model_checkpoint'
        self.epoch = epoch
        self.weight = weights
        self.model = FaceClassify().to(device)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(self.weight).cuda())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.dataset = MakeData()
        self.train_losses = []
        self.valid_losses = []
        self.is_save = is_save
        self.loss = 100000
        self.early_stop = early_stop
        self.best_model_dict = {}

    def plot_loss(self):
        plt.plot(self.train_losses, label='Train loss')
        plt.plot(self.valid_losses, label='Valid loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend(frameon=False)
        plt.show()

    def predict(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = self.model(images)
                outputs = torch.squeeze(outputs, 1)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
            print('Test Accuracy: {} %'.format(acc))

    def save_model(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_{}.ckpt'.format(str(self.loss))))

    def train(self):
        anchor = 0
        train_loader, valid_loader = MakeData().make_loader()
        print('Start train')
        for epoch in range(1, self.epoch + 1):
            train_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for data, target in train_loader:
                target = target.type(torch.LongTensor)
                data = data.to(device)
                target = target.to(device)

                self.optimizer.zero_grad()
                output = self.model(data)
                output = torch.squeeze(output, 1)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)

            self.model.eval()
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device).long()
                output = self.model(data)
                output = torch.squeeze(output, 1)
                loss = self.criterion(output, target)
                valid_loss += loss.item() * data.size(0)

            if self.early_stop > 0:
                if self.loss > valid_loss:
                    self.loss = valid_loss
                    anchor = epoch
                    self.best_model_dict = self.model.state_dict()
                else:
                    if (epoch - anchor) == self.early_stop:
                        model_dict = self.model.state_dict()
                        pretrained_dict = {k: v for k, v in self.best_model_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        self.model.load_state_dict(model_dict)
                        print('Model drop at {} epoch'.format(epoch))
                        break
            len_train, len_valid = self.dataset.get_len()
            train_loss = train_loss / len_train
            valid_loss = valid_loss / len_valid
            self.train_losses.append(train_loss)
            self.valid_losses.append(valid_loss)

            print('Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
                epoch, train_loss, valid_loss))
        self.predict(valid_loader)
        self.plot_loss()
        if self.is_save:
            self.save_model()


def evaluate_model(path2model, path2test, label):
    model = FaceClassify()
    model.load_state_dict(torch.load(path2model))
    model.to(device)
    print('Load model_checkpoint done')
    test = create_data(path2data=path2test, labels=label)
    test_data = FaceDataset(test, transform=normalize)
    model.eval()
    print('Evaluating.....')
    with torch.no_grad():
        correct = np.zeros(10)
        total = np.ones(10)
        for feature, label in test_data:
            feature = feature.to(device)
            predicted = model(feature)
            predicted = torch.squeeze(predicted, 1)
            predicted = torch.argmax(predicted.data, 1)
            predicted = predicted.data[0].cpu().numpy()
            print(predicted, label)
            total[label] += 1
            correct[label] += (predicted == label)
        acc = 100*correct/total
    print('Acc: {}%'.format(acc))


def main():
    args = parse_args()

    if args.is_train == 'True':
        TrainModel(is_save=args.save, epoch=args.epoch, early_stop=args.early_stop).train()
    else:
        evaluate_model(path2model=args.model, path2test=args.test_data, label=args.test_label)


if __name__ == '__main__':
    main()