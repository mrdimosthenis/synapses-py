from typing import List
from functional import seq

from synapses_py.model.net_elems.layer import layer
from synapses_py.model.net_elems.neuron import neuron_serialized

LayerSerialized = List[neuron_serialized.NeuronSerialized]


def serialized(layer_val: layer.Layer) -> LayerSerialized:
    return seq(layer_val) \
        .map(lambda x: neuron_serialized.serialized(x)) \
        .to_list()


def deserialized(layer_serialized: LayerSerialized) -> layer.Layer:
    return seq(layer_serialized) \
        .map(lambda x: neuron_serialized.deserialized(x))
