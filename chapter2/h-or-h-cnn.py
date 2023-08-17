import urllib.request
import zipfile
import torch
import torchvision
from torchvision.io import read_image
import pandas as pd
import torch.nn as nn
import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as fn
from torch.utils.data import DataLoader, Dataset

# training_url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
# training_file_name = "horse-or-human.zip"
# training_dir = './horse-or-human/training/'
# urllib.request.urlretrieve(training_url, training_file_name)

# zip_ref = zipfile.ZipFile(training_file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()

# validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"
# validation_file_name = "validation-horse-or-human.zip"
# validation_dir = 'horse-or-human/validation/'
# urllib.request.urlretrieve(validation_url, validation_file_name)

# zip_ref = zipfile.ZipFile(validation_file_name, 'r')
# zip_ref.extractall(validation_dir)
# zip_ref.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

def makeDataframe(path, label):
    file_names = []    
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            file_names.append(filename)

    data = pd.DataFrame()
    data['file'] = file_names
    data['label'] = label
    return data

def makeDataset(type):
    data_set_horses = makeDataframe(f'./horse-or-human/{type}/horses/', 0)
    data_set_humans = makeDataframe(f'./horse-or-human/{type}/humans/', 1)
    dataset = pd.concat([data_set_horses, data_set_humans])
    dataset = dataset.sample(frac=1)
    return dataset   

training_dataset = makeDataset(type='training')
validation_dataset = makeDataset(type='validation')

class HorseHumanDataset(Dataset):
    def __init__(self, dataframe, transform=None, target_transform=None, type=None):
        self.labels = dataframe['label'].values
        self.img_path = dataframe['file'].values
        self.img_dirs = [f'./horse-or-human/{type}/horses/', f'./horse-or-human/{type}/humans/']
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        img_path = f"{self.img_dirs[label]}{self.img_path[idx]}"
        image = read_image(img_path, torchvision.io.ImageReadMode.RGB).type(torch.float)
        image /= 255.0
        return image, label

training_data = HorseHumanDataset(
    dataframe=training_dataset,
    transform=data_transform,
    type='training'
)

validation_data = HorseHumanDataset(
    dataframe=validation_dataset,
    transform=data_transform,
    type='validation'
)

train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=64, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 1)


    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.pool(torch.relu(self.conv4(x)))
        x = self.pool(torch.relu(self.conv5(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = Model().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

epochs = 15
for epoch in range(epochs):
    model.train()
    for image, label in train_loader:
        print("============================Image============================")
        print(image)
        print("============================Label============================")
        print(label)
        image = image.to(device)
        label = label.to(device)
        label = label.unsqueeze(1).float()

        optimizer.zero_grad()
        outputs = model(image)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)

            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            print("Predictions:", predicted)
            total_samples += label.size(0)
            total_correct += (predicted == label).sum().item()
        
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.4f}")
        if accuracy > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            break

print("Training finished!")

# Printing a few predictions
model.eval()
with torch.no_grad():
    image, label = next(iter(test_loader))
    image = image.to(device)
    label = label.to(device)

    label = label.unsqueeze(1).float()

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    print("Predictions:", predicted)
    print("Actual label:", label)

    acc = (predicted == label).sum().item()*1.0/len(label)
    print(f'acc = {acc}')