from typing import Callable

from functional import seq
from functional.pipeline import Sequence

from synapses_py.model import utilities
from synapses_py.model.net_elems import layer
from synapses_py.model.net_elems.activation import activation
from synapses_py.model.net_elems.layer import layer

Network = Sequence


def realize(network_val: Network) -> Network:
    return network_val \
        .map(lambda x: layer.realize(x)) \
        .cache()


# public
def init(layer_sizes: Sequence,
         activation_f: Callable[[int], activation.Activation],
         weight_init_f: Callable[[int], float]
         ) -> Network:
    return layer_sizes.zip(layer_sizes.tail()) \
        .zip_with_index() \
        .map(lambda t:
             layer.init(
                 t[0][0],
                 t[0][1],
                 activation_f(t[1]),
                 lambda: lambda: weight_init_f(t[1])))


# public
def output(input_val: Sequence,
           in_parallel: bool,
           network_val: Network
           ) -> Sequence:
    return network_val.fold_left(
        input_val,
        lambda acc, x: layer.output(acc, in_parallel, x)
    )


def fed_forward_acc_f(already_fed: Sequence,
                      in_parallel: bool,
                      next_layer: layer.Layer
                      ) -> Sequence:
    (errors_val, layer_val) = already_fed.head()
    next_input = layer.output(errors_val, in_parallel, layer_val)
    return utilities \
        .lazy_cons((next_input, next_layer), already_fed)


def fed_forward(input_val: Sequence,
                in_parallel: bool,
                network: Network
                ) -> Sequence:
    init_feed = seq([(input_val, network.head())])
    return network \
        .tail() \
        .fold_left(init_feed, lambda acc, x: fed_forward_acc_f(acc, in_parallel, x))


def back_propagated_acc_f(learning_rate: float,
                          in_parallel: bool,
                          errors_with_already_propagated: Sequence,  # TODO: this is a tuple
                          input_with_layer: (Sequence, layer.Layer)
                          ) -> (Sequence, Sequence):
    (errors_val, already_propagated) = errors_with_already_propagated
    (last_input, last_layer) = input_with_layer
    last_output_with_errors = layer \
        .output(last_input, in_parallel, last_layer) \
        .zip(errors_val)
    (next_errors, propagated_layer) = layer.back_propagated(
        learning_rate,
        last_input,
        last_output_with_errors,
        in_parallel,
        last_layer
    )
    next_already_propagated = utilities \
        .lazy_cons(propagated_layer, already_propagated)
    return next_errors, next_already_propagated


def back_propagated(learning_rate: float,
                    expected_output: Sequence,
                    reversed_inputs_with_layers: Sequence,
                    in_parallel: bool
                    ) -> (Sequence, Network):
    (last_input, last_layer) = reversed_inputs_with_layers.head()
    output_val = layer.output(last_input, in_parallel, last_layer)
    errors_val = output_val \
        .zip(expected_output) \
        .map(lambda t: t[0] - t[1])
    output_with_errors = output_val.zip(errors_val)
    (init_errors, first_propagated) = layer.back_propagated(
        learning_rate,
        last_input,
        output_with_errors,
        in_parallel,
        last_layer
    )
    init_acc = (init_errors, seq([first_propagated]))
    return reversed_inputs_with_layers \
        .tail() \
        .fold_left(init_acc,
                   lambda acc, x:
                   back_propagated_acc_f(learning_rate, in_parallel, acc, x))


def errors_with_fit_net(learning_rate: float,
                        input_val: Sequence,
                        expected_output: Sequence,
                        in_parallel: bool,
                        network: Network
                        ) -> (Sequence, Network):
    return back_propagated(
        learning_rate,
        expected_output,
        fed_forward(input_val, in_parallel, network),
        in_parallel
    )


# public
def errors(learning_rate: float,
           input_val: Sequence,
           expected_output: Sequence,
           in_parallel: bool,
           network: Network
           ) -> Sequence:
    return errors_with_fit_net(
        learning_rate,
        input_val,
        expected_output,
        in_parallel,
        network
    )[0]


# TODO: fit returns Network
# TODO: refactor common code between the above and below function in all languages

# public
def fit(learning_rate: float,
        input_val: Sequence,
        expected_output: Sequence,
        in_parallel: bool,
        network: Network
        ) -> Sequence:
    return errors_with_fit_net(
        learning_rate,
        input_val,
        expected_output,
        in_parallel,
        network
    )[1]
