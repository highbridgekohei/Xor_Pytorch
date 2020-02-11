import torch  # type: ignore
import torch.nn as nn  # type: ignore
import torch.optim as optim  # type: ignore
import numpy as np  # type: ignore
from typing import Callable, Any, List


class Net(nn.Module):
    def __init__(
        self,
        structure: List[int],
        initializer: Callable[..., Any] = lambda w: nn.init.uniform_(w, -1.0, 1.0),
    ):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.layer_num: int = len(structure)
        for i in range(1, self.layer_num):
            setattr(self, "fc%d" % i, nn.Linear(structure[i - 1], structure[i]))
            initializer(getattr(self, "fc%d" % i).weight)

    def forward(self, x: torch.Tensor):
        for i in range(1, self.layer_num):
            x = torch.sigmoid(getattr(self, "fc%d" % i)(x))
        return x


# ----------- Selecting Optimizer -----------
if __name__ == "__main__":
    TRAINING_DATA: torch.Tensor = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    TEACHING_DATA: torch.Tensor = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    EPOCH: int = 10000
    error_boundary: float = 1e-3
    initializer: Callable[..., Any] = lambda weights: nn.init.uniform_(
        weights, -0.5, 0.5
    )
    net = Net([2, 2, 1], initializer)
    criterion = nn.MSELoss()
    learning_rate: float = 0.5
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
