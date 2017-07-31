import unittest

from model_training import load_data, build_network
from model_inference import retrieve_model


class TestMethods(unittest.TestCase):
    def test_load_data(self):
        expected_output = (55000, 28, 28, 1)
        actual_output = load_data()[0].shape
        self.assertEqual(expected_output, actual_output)

    def test_build_network(self):
        expected_output = 'FullyConnected_2/Softmax:0'
        actual_output = build_network().name
        self.assertEqual(expected_output, actual_output)

    def test_retrieve_model(self):
        expected_output = 10
        actual_output = retrieve_model().targets[0].shape.dims[1].value
        self.assertEqual(expected_output, actual_output)


if __name__ == '__main__':
    unittest.main()
