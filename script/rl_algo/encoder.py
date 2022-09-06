import torch.nn as nn
import torch 

class CNNEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        modules = []
        
        # Build Encoder
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels = 3, out_channels=32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )
        )

        self.net = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(64*8*6, 32)
    
    def forward(self, x):
        x = self.net(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_mu(x)
        return x