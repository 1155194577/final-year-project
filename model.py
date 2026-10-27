import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)

# # Example usage:
# layer = LinearLayer(in_features=input_size, out_features=output_size)
# print(layer)