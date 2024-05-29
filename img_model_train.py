import pickle

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from torchvision.models import resnet18
from PIL import Image
import os

from tqdm import tqdm
from transformers import ViTModel
from functools import partial
import torch.nn.functional as F


data_folder = 'D:\ei\Sentiment_Analysis_Imdb-master\cnn+vit\cnn+vit\image_test'
batch_size = 32

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])



class CustomDataset(Dataset):
    def __init__(self, address,label, transform=None):
        self.root_dir = ''
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图像尺寸为 224x224
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
        self.file_list = address
        self.labels = label




    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def load_dataset(train_batch_size, test_batch_size,workers=0):
    data = pd.read_csv('cv_address.csv',encoding='gbk', engine='python')
    len1 = int(len(list(data['labels'])))
    labels = list(data['labels'])[0:len1]
    sentences = list(data['sentences'])[0:len1]
    # split train_set and test_set
    tr_sen, te_sen, tr_lab, te_lab = train_test_split(sentences, labels, train_size=0.8)
    # Dataset
    train_set = CustomDataset(tr_sen, tr_lab)
    test_set = CustomDataset(te_sen, te_lab)
    # DataLoader
    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=workers,
                               pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=workers,
                              pin_memory=True)
    return train_loader, test_loader

# train_dataset = CustomDataset(os.path.join(data_folder, 'train'), transform=data_transforms)
# test_dataset = CustomDataset(os.path.join(data_folder, 'test'), transform=data_transforms)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

train_dataloader, test_dataloader = load_dataset(batch_size,batch_size)

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(64 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x


local_model_path = 'D:\ei\Sentiment_Analysis_Imdb-master\cnn+vit\cnn+vit\weight'
vit_model = ViTModel.from_pretrained(local_model_path)
vit_model.training = True

vit_model.config.num_labels = 2
vit_model.classifier = nn.Linear(vit_model.config.hidden_size, vit_model.config.num_labels)


class CombinedModel(nn.Module):
    def __init__(self, cnn_model, vit_model, num_classes):
        super(CombinedModel, self).__init__()
        self.cnn_model = cnn_model
        self.vit_model = vit_model
        self.fc = nn.Linear(vit_model.config.hidden_size + 50176, 1536)
        self.adjust = nn.Linear(1536, num_classes)

    def forward(self, x):
        cnn_features = self.cnn_model(x)
        vit_features = self.vit_model(x).last_hidden_state[:,0,:]
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        feature = self.fc(combined_features)
        x = self.adjust(feature)
        return x

num_classes = 2
cnn_model = ConvNet(num_classes)
combined_model = CombinedModel(cnn_model, vit_model, 2)


num_epochs = 50
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(combined_model.parameters(), lr=learning_rate)
combined_model = combined_model.to(device)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(dataloader, ascii='>='):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        running_loss += loss.item()

    train_loss = running_loss / len(dataloader)
    train_acc = correct / total

    return train_loss, train_acc

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, ascii='>='):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

    val_loss = running_loss / len(dataloader)
    val_acc = correct / total

    return val_loss, val_acc

best_loss, best_acc = 0, 0
l_acc, l_trloss, l_teloss, l_epo = [], [], [], []
for epoch in range(num_epochs):
    train_loss, train_acc = train(combined_model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(combined_model, test_dataloader, criterion, device)
    l_acc.append(train_acc)
    l_trloss.append(train_loss)
    l_teloss.append(val_loss)
    l_epo.append(epoch)
    if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
        best_acc, best_loss = val_acc, val_loss
        with open('cv_combine.pkl', "wb") as file:
            pickle.dump(combined_model, file)

    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    print("--------------------------")
plt.plot(l_epo, l_acc)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('acc.png')

plt.plot(l_epo, l_teloss)
plt.ylabel('test-loss')
plt.xlabel('epoch')
plt.savefig('teloss.png')

plt.plot(l_epo, l_trloss)
plt.ylabel('train-loss')
plt.xlabel('epoch')
plt.savefig('trloss.png')
