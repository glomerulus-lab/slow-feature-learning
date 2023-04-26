import timeit
import random
import unittest

class BenchmarkTest(unittest.TestCase):

    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_my_function_benchmark(self):
        def my_function():
            pass

        num_runs = 1000
        times = timeit.repeat(my_function, number=1, repeat=num_runs)
        print(f"AVG EXECUTION TIME: {sum(times)/len(times)}")

