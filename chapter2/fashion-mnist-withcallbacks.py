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
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=1)
        return x

model = Model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0012)

epochs = 50
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
        if accuracy > 0.95:
            print("\nReached 95% accuracy so cancelling training!")
            break

print("Training finished!")