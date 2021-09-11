from typing import List, Dict
from dataclasses import dataclass
from functional import seq
from synapses_py.model.net_elems.activation import activation_serialized
from synapses_py.model.net_elems.neuron import neuron


@dataclass(frozen=True)
class NeuronSerialized:
    activationF: activation_serialized.ActivationSerialized
    weights: List[float]


def serialized(neuron: neuron.Neuron) -> NeuronSerialized:
    return NeuronSerialized(
        activation_serialized.serialized(neuron.activation_f),
        neuron.weights.to_list()
    )


def deserialized(neuron_serialized: Dict) -> neuron.Neuron:
    return neuron.Neuron(
        activation_serialized.deserialized(neuron_serialized['activationF']),
        seq(neuron_serialized['weights'])
    )
