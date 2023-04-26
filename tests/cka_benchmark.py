import torch

class CKABenchmarkTest(BenchmarkTest, my_function):
    def setUp(self):
        self.y = torch.randn(16384, 1)
        self.phi = torch.randn(16384, 2048)

    def test_function(self):
        def function(y, phi):
            return my_function(y, phi)
