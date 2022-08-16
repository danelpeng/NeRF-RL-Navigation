import torch
import unittest
from vae import VAE
from torchsummary import summary

class TestVAE(unittest.TestCase):
    def setUp(self) -> None:
        self.model = VAE(3,32)
    
    def test_summary(self):
        print(summary(self.model, (3, 128, 96), device='cpu'))
    
    def test_forward(self):
        x = torch.randn(16, 3, 128, 96)
        y = self.model(x)
        print("Model Output size:",y[0].size())
    
    def test_loss(self):
        x = torch.randn(16, 3, 128, 96)

        result = self.model(x)
        loss = self.model.loss_function(*result, M_N = 0.005)
        print(loss)

if __name__ == '__main__':
    unittest.main()