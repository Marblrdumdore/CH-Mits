import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from transformers import ViTForImageClassification


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []
        self.labels = []

        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for file_name in os.listdir(label_dir):
                    file_path = os.path.join(label_dir, file_name)
                    self.file_list.append(file_path)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        image = Image.open(file_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

data_folder = '/hy-tmp/cnn+vit/image_test'  # Update this path
train_dataset = CustomDataset(os.path.join(data_folder, 'train'), transform=data_transforms)
test_dataset = CustomDataset(os.path.join(data_folder, 'test'), transform=data_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


local_model_path = '/hy-tmp/cnn+vit/weight'
vit_model = ViTForImageClassification.from_pretrained(local_model_path, num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vit_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_model.parameters(), lr=0.001)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    print(f'Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images).logits
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    print(f'Val Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')


num_epochs = 10
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    train(vit_model, train_loader, criterion, optimizer, device)
    evaluate(vit_model, test_loader, criterion, device)