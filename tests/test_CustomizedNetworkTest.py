import unittest
from random import seed, random

from synapses_py import fun, net


def activation_f(layer_index: int) -> fun.Fun:
    if layer_index == 0:
        return fun.SIGMOID
    elif layer_index == 1:
        return fun.IDENTITY
    elif layer_index == 2:
        return fun.LEAKY_RE_LU
    else:
        return fun.TANH


def weight_init_f(_layer_index: int) -> float:
    return 1.0 - 2.0 * random()


class TestCustomizedNetworkTest(unittest.TestCase):
    seed(1000)

    layers = [4, 6, 5, 3]

    justCreatedNeuralNetwork = net.Net(
        layers,
        activation_f=activation_f,
        weight_init_f=weight_init_f
    )
    justCreatedNeuralNetworkJson = justCreatedNeuralNetwork.json()

    neuralNetworkJsonFile = open("resources/network.json", "r")
    neuralNetworkJson = neuralNetworkJsonFile.read()
    neuralNetworkSvgFile = open("resources/drawing.svg", "r")
    neuralNetworkSvg = neuralNetworkSvgFile.read()
    expected_net_json = net.Net(json=justCreatedNeuralNetworkJson).json()
    neuralNetwork = net.Net(json=neuralNetworkJson)
    expected_svg = neuralNetwork.svg()
    inputValues = [1.0, 0.5625, 0.511111, 0.47619]
    expectedOutput = [0.4, 0.05, 0.2]
    prediction = neuralNetwork.par_predict(inputValues)
    expected_normal_errors = neuralNetwork.errors(inputValues, expectedOutput, True)
    expected_zero_errors = neuralNetwork.errors(inputValues, prediction, True)
    neuralNetwork.par_fit(0.01, inputValues, expectedOutput)
    expected_fit_prediction = neuralNetwork.par_predict(inputValues)

    def test_neuralNetworkOfToJson(self):
        net_json = net.Net(json=self.justCreatedNeuralNetworkJson)
        self.assertEqual(
            self.justCreatedNeuralNetworkJson,
            self.expected_net_json
        )

    def test_neuralNetworkPrediction(self):
        self.assertEqual(
            [-0.013959435951885571, -0.16770539176070537, 0.6127887629040738],
            self.prediction
        )

    def test_neuralNetworkNormalErrors(self):
        self.assertEqual(
            [-0.18229373795952453, -0.10254022760223255, -0.09317233470223055, -0.086806455078946],
            self.expected_normal_errors
        )

    def test_neuralNetworkZeroErrors(self):
        self.assertEqual(
            [0.0, 0.0, 0.0, 0.0],
            self.expected_zero_errors
        )

    def test_neuralNetworkOfToSvg(self):
        self.assertEqual(
            self.neuralNetworkSvg,
            self.expected_svg
        )

    def test_fitNeuralNetworkPrediction(self):
        self.assertEqual(
            [-0.006109464554743645, -0.1770428172237149, 0.6087944183600162],
            self.expected_fit_prediction
        )

    neuralNetworkJsonFile.close()
    neuralNetworkSvgFile.close()


if __name__ == '__main__':
    unittest.main()
