from random import seed, random
from typing import List, Optional, Callable

from functional import seq

from synapses_py.model import draw
from synapses_py.model.net_elems import network
from synapses_py.model.net_elems.network import network, network_serialized
from synapses_py import fun


def throw_if_input_not_match(network_val: network.Network,
                             input_values: List[float]) -> None:
    num_of_input_vals = len(input_values)
    input_layer_size = network_val.head().head().weights.size() - 1
    error_msg = 'the number of input values (' + \
                str(num_of_input_vals) + \
                ') does not match the size of the input layer (' + \
                str(input_layer_size) + \
                ')'
    if num_of_input_vals != input_layer_size:
        raise Exception(error_msg)


def throw_if_expected_not_match(network_val: network.Network,
                                expected_output: List[float]) -> None:
    num_of_expected_vals = len(expected_output)
    output_layer_size = network_val.last().size()
    error_msg = 'the number of expected values (' + \
                str(num_of_expected_vals) + \
                ') does not match the size of the output layer (' + \
                str(output_layer_size) + \
                ')'
    if num_of_expected_vals != output_layer_size:
        raise Exception(error_msg)


def seed_init_network(maybe_seed: Optional[int],
                      layers: List[int]
                      ) -> network.Network:
    layer_sizes = seq(layers)
    if maybe_seed is not None:
        seed(maybe_seed)
    return network.init(
        layer_sizes,
        lambda _: fun.SIGMOID,
        lambda _: 1.0 - 2.0 * random()
    )


def init(layers: List[int]) -> network.Network:
    return seed_init_network(None, layers)


def init_with_seed(layers: List[int], seed_val: int) -> network.Network:
    return seed_init_network(seed_val, layers)


def custom_init(layers: List[int],
                activation_f: Callable[[int], fun.Fun],
                weight_init_f: Callable[[int], float]
                ) -> network.Network:
    layer_sizes = seq(layers)
    return network.init(
        layer_sizes,
        activation_f,
        weight_init_f
    )


class Net:

    def __init__(self,
                 layers: Optional[List[int]] = None,
                 seed: Optional[int] = None,
                 activation_f: Optional[Callable[[int], fun.Fun]] = None,
                 weight_init_f: Optional[Callable[[int], float]] = None,
                 json: Optional[str] = None):
        if activation_f is not None:
            contents = custom_init(layers, activation_f, weight_init_f)
        elif seed is not None:
            contents = init_with_seed(layers, seed)
        elif json is not None:
            contents = network_serialized.of_json(json)
        else:
            contents = init(layers)
        self.__contents = network.realize(contents)

    def predict(self, input_values: List[float]) -> List[float]:
        throw_if_input_not_match(self.__contents, input_values)
        input_val = seq(input_values)
        return network \
            .output(input_val, False, self.__contents) \
            .to_list()

    def par_predict(self, input_values: List[float]) -> List[float]:
        throw_if_input_not_match(self.__contents, input_values)
        input_val = seq(input_values)
        return network \
            .output(input_val, True, self.__contents) \
            .to_list()

    def errors(self,
               input_values: List[float],
               expected_output: List[float],
               in_parallel: bool
               ) -> List[float]:
        throw_if_input_not_match(self.__contents, input_values)
        throw_if_expected_not_match(self.__contents, expected_output)
        input_val = seq(input_values)
        expected_val = seq(expected_output)
        return network \
            .errors(0.0,  # learning rate do not affect the error calculation
                    input_val,
                    expected_val,
                    in_parallel,
                    self.__contents) \
            .to_list()

    def fit(self,
            learning_rate: float,
            input_values: List[float],
            expected_output: List[float]
            ) -> None:
        throw_if_input_not_match(self.__contents, input_values)
        throw_if_expected_not_match(self.__contents, expected_output)
        input_val = seq(input_values)
        expected_val = seq(expected_output)
        self.__contents = network.realize(
            network.fit(
                learning_rate,
                input_val,
                expected_val,
                False,
                self.__contents
            )
        )

    def par_fit(self,
                learning_rate: float,
                input_values: List[float],
                expected_output: List[float]
                ) -> None:
        throw_if_input_not_match(self.__contents, input_values)
        throw_if_expected_not_match(self.__contents, expected_output)
        input_val = seq(input_values)
        expected_val = seq(expected_output)
        self.__contents = network.realize(
            network.fit(
                learning_rate,
                input_val,
                expected_val,
                True,
                self.__contents
            )
        )

    def json(self) -> str:
        return network_serialized.to_json(self.__contents)

    def svg(self) -> str:
        return draw.network_svg(self.__contents)
