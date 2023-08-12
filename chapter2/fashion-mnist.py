import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=data_transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=data_transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.softmax(self.fc4(x), dim=1)
        return x
    
model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)

epochs = 20

for epoch in range(epochs):
    model.train()
    for image, label in train_loader:

        image = image.to(device)
        label = label.to(device)

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
            total_samples += label.size(0)
            total_correct += (predicted == label).sum().item()
        
        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {accuracy:.4f}")

print("Training finished!")

# Printing a few predictions
model.eval()
with torch.no_grad():
    image, label = next(iter(test_loader))

    image = image.to(device)
    label = label.to(device)

    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    print("Predictions:", predicted)
    print("Actual label:", label)

    acc = (predicted == label).sum().item()*1.0/len(label)
    print(f'acc = {acc}')

        
