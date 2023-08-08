import torch
import torch.nn as nn
import numpy as np

# Check if a GPU is available
# cuda rather than cuda:0 => The device index is only useful when you have multiple gpus.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.l0 = nn.Linear(1, 1)

    def forward(self, x):
        return self.l0(x)

model = Model().to(device)  # Move the model to GPU if available
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)

xs_tensor = torch.tensor(xs[:, np.newaxis], dtype=torch.float32).to(device)
ys_tensor = torch.tensor(ys[:, np.newaxis], dtype=torch.float32).to(device)

epochs = 500
for epoch in range(epochs):
    # Forward pass
    y_pred = model(xs_tensor)
    loss = criterion(y_pred, ys_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Prediction
input_tensor = torch.tensor([[10.0]], dtype=torch.float32).to(device)
prediction = model(input_tensor).item()
print(prediction)

# Print learned weights
print("Here is what I learned: {}".format(list(model.parameters())))
