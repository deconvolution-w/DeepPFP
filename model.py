import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self, squeeze_ratio):
        super(Net, self).__init__()
        self.hidden_dim = int(5120 / squeeze_ratio)
        self.base = nn.Sequential(
            nn.Linear(5120, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.fc(x)
        return x

                   
                    
if __name__ == '__main__':
    model = Net(16)
    print(model)
    x = torch.randn((5,10))