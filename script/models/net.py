import torch.nn as nn

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        num_layers=2,
        hidden_layer_size=512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = hidden_layer_size

        layers = []
        for i in range(num_layers):
            input_dim = hidden_layer_size if i > 0 else self.input_dim
            layers.append(nn.Linear(input_dim, hidden_layer_size))
            layers.append(nn.ReLU())
        
        self.mlp = nn.Sequential(*layers)

    def forward(self,x):
        return self.mlp(x)