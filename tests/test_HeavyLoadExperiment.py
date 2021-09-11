import unittest
import time

from random import random
from typing import List

from synapses_py import net, fun

first_layer_size = 1000
last_layer_size = 2
layers = [first_layer_size, 1500, 3000, 10, last_layer_size]


def activation_f(layer_index: int) -> fun.Fun:
    if layer_index == 0:
        return fun.LEAKY_RE_LU
    elif layer_index == 1:
        return fun.IDENTITY
    elif layer_index == 2:
        return fun.LEAKY_RE_LU
    elif layer_index == 3:
        return fun.IDENTITY


def weight_init_f(_layer_index: int) -> float:
    return 1.0 - 2.0 * random()


def random_input_values() -> List[float]:
    return [random() for _ in range(first_layer_size)]


def random_output_values() -> List[float]:
    return [random() for _ in range(last_layer_size)]


def current_milli_time():
    return round(time.time() * 1000)


class TestHeavyLoadExperiment(unittest.TestCase):

    @unittest.skip
    def test_heavy_load_experiment(self):
        start_millis = current_milli_time()
        neural_network = net.Net(
            layers,
            activation_f=activation_f,
            weight_init_f=weight_init_f
        )
        for _ in range(3):
            neural_network.fit(
                0.1,
                random_input_values(),
                random_output_values()
            )
        last_prediction_size = len(
            neural_network.predict(random_input_values())
        )
        end_millis = current_milli_time()
        duration = str(end_millis - start_millis)
        print(duration + ': the duration of SERIAL experiment')
        self.assertEqual(last_prediction_size, last_layer_size)

    @unittest.skip
    def test_heavy_load_experiment_in_parallel(self):
        start_millis = current_milli_time()
        neural_network = net.Net(
            layers,
            activation_f=activation_f,
            weight_init_f=weight_init_f
        )
        for _ in range(3):
            neural_network.par_fit(
                0.1,
                random_input_values(),
                random_output_values()
            )
        last_prediction_size = len(
            neural_network.par_predict(random_input_values())
        )
        end_millis = current_milli_time()
        duration = str(end_millis - start_millis)
        print(duration + ': the duration of PARALLEL experiment')
        self.assertEqual(last_prediction_size, last_layer_size)


if __name__ == '__main__':
    unittest.main()
