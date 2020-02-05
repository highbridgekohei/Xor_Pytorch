import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self, structure, range_value):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(structure[0], structure[1])
        nn.init.uniform_(self.fc1.weight, -range_value, range_value)
        self.fc2 = nn.Linear(structure[1], structure[2])
        nn.init.uniform_(self.fc2.weight, -range_value, range_value)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# ----------- Selecting Optimizer -----------
if __name__ == "__main__":
    import torch.optim as optim

    TRAINING_DATA = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    TEACHING_DATA = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    EPOCH = 10000
    error_boundary = 1e-3
    vr = 0.5
    learning_rate = 0.5
    net = Net([2, 2, 1], vr)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    #     optimizer = optim.Adam(net.parameters(), lr = 0.0001)
    training_data_indexes = np.arange(len(TRAINING_DATA))
    Loss = []
    for epoch in range(EPOCH):
        error = []
        for data_index in training_data_indexes:
            optimizer.zero_grad()
            output = net(TRAINING_DATA[data_index])
            loss = criterion(output, TEACHING_DATA[data_index])
            loss.backward()
            optimizer.step()
            error.append(float(loss.mean()))
        Loss.append(np.mean(error))
        if Loss[epoch] < error_boundary:
            print(epoch, Loss[epoch])
            print("----- End Learning -----")
            break
        if epoch % 1000 == 0:
            print(epoch, Loss[epoch])
    for training_data in TRAINING_DATA:
        output = net(training_data)
        print(output)
