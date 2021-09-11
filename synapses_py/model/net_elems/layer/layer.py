from itertools import count
from typing import Callable, Tuple, Any
from functional import seq, pseq
from functional.pipeline import Sequence

from synapses_py.model import utilities
from synapses_py.model.net_elems.activation import activation
from synapses_py.model.net_elems.neuron import neuron

Layer = Sequence


def realize(layer_val: Layer) -> Layer:
    return layer_val \
        .map(lambda x: neuron.realize(x)) \
        .cache()


def pmap(sequence: Sequence, f: Callable[[Any], Any]) -> Sequence:
    return seq(
        pseq(sequence).map(lambda x: f(x))
    )


def init(input_size: int,
         output_size: int,
         activation_f: activation.Activation,
         weight_init_f: Callable[[], Callable[[], float]]
         ) -> Layer:
    return seq \
        .range(output_size) \
        .map(lambda _:
             neuron.init(input_size,
                         activation_f,
                         weight_init_f()))


def output(input_val: Sequence, in_parallel: bool, layer_val: Layer) -> Sequence:
    if in_parallel:
        return pmap(layer_val, lambda x: neuron.output(input_val, x))
    else:
        return layer_val.map(lambda x: neuron.output(input_val, x))


def back_propagated(learning_rate: float,
                    input_val: Sequence,
                    output_with_errors: Sequence,
                    in_parallel: bool,
                    layer: Layer
                    ) -> Tuple[Sequence, Layer]:
    if in_parallel:
        errors_multi_with_new_neurons = pmap(
            output_with_errors.zip(layer),
            lambda t: neuron.back_propagated(learning_rate, input_val, t[0], t[1])
        )
    else:
        errors_multi_with_new_neurons = output_with_errors \
            .zip(layer) \
            .map(lambda t: neuron.back_propagated(learning_rate, input_val, t[0], t[1]))
    (errors_multi, new_neurons) = utilities \
        .lazy_unzip(errors_multi_with_new_neurons)
    in_errors = errors_multi.fold_left(
        seq(count(0)).map(lambda _: 0.0),
        lambda acc, x: acc.zip(x).map(lambda t: t[0] + t[1])
    )
    return in_errors, new_neurons
