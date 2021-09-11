import json
from typing import List

from functional import seq

from synapses_py.model import utilities
from synapses_py.model.net_elems.layer import layer_serialized
from synapses_py.model.net_elems.network import network

NetworkSerialized = List[layer_serialized.LayerSerialized]


def serialized(network_val: network.Network) -> NetworkSerialized:
    return network_val \
        .map(lambda x: layer_serialized.serialized(x)) \
        .to_list()


# public
def to_json(network: network.Network) -> str:
    return json.dumps(
        serialized(network),
        separators=(',', ':'),
        cls=utilities.EnhancedJSONEncoder
    )


def deserialized(network_serialized: NetworkSerialized) -> network.Network:
    return seq(network_serialized).map(lambda x: layer_serialized.deserialized(x))


# public
def of_json(s: str) -> network.Network:
    return deserialized(json.loads(s))
