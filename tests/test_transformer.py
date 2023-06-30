import unittest
from transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.transformer = Transformer(d_model=16, nhead=2, num_encoder_layers=2,
                                       num_decoder_layers=2, dim_feedforward=128)

    def test_model_creation(self):
        self.assertIsNotNone(self.transformer, "Model creation failed.")

    def test_model_forward_pass(self):
        # Assuming your input is of shape (batch_size, sequence_length)
        input_tensor = torch.rand(8, 10, 16)
        output = self.transformer(input_tensor)
        self.assertEqual(output.shape, (8, 10, 16), "Forward pass output shape is incorrect.")

    def test_model_backward_pass(self):
        # Assuming your input is of shape (batch_size, sequence_length)
        input_tensor = torch.rand(8, 10, 16)
        output = self.transformer(input_tensor)
        output.mean().backward()  # Backward pass
        for name, param in self.transformer.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient is None for parameter {name}.")

if __name__ == '__main__':
    unittest.main()
