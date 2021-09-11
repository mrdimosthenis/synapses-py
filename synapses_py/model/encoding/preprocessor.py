import json
from typing import Dict

from functional import seq
from functional.pipeline import Sequence

from synapses_py.model import utilities
from synapses_py.model.encoding.attribute import attribute, attribute_serialized

Preprocessor = Sequence


def realize(preprocessor: Preprocessor) -> Preprocessor:
    return preprocessor \
        .map(lambda x: attribute.realize(x)) \
        .cache()


def updated(datapoint: Dict[str, str],
            preprocessor: Preprocessor) -> Preprocessor:
    return preprocessor \
        .map(lambda x: attribute.updated(datapoint, x))


def init_f(key: str,
           is_discrete: bool,
           dataset_head: Sequence
           ) -> attribute.Attribute:
    if is_discrete:
        return attribute.DiscreteAttribute(
            key,
            seq([dataset_head[key]])
        )
    else:
        v = attribute.parse(dataset_head[key])
        return attribute.ContinuousAttribute(
            key,
            v,
            v
        )


# public
def init(keys_with_flags: Sequence,
         dataset: Sequence
         ) -> Preprocessor:
    dataset_head = dataset.head()
    dataset_tail = dataset.tail()
    init_preprocessor = keys_with_flags \
        .map(lambda x: init_f(x[0], x[1], dataset_head))
    return realize(
        dataset_tail.fold_left(init_preprocessor, lambda acc, x: updated(x, acc))
    )


# public
def encode(datapoint: Dict[str, str],
           preprocessor: Preprocessor
           ) -> Sequence:
    return preprocessor \
        .flat_map(lambda x: attribute.encode(datapoint[x.key], x))


def decode_acc_f(acc: (Sequence, Sequence),
                 attr: attribute.Attribute
                 ) -> (Sequence, Sequence):
    (unprocessed_floats, processed_ks_vs) = acc
    if isinstance(attr, attribute.DiscreteAttribute):
        split_index = attr.values.size()
    elif isinstance(attr, attribute.ContinuousAttribute):
        split_index = 1
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')
    (encoded_values, next_floats) = utilities \
        .lazy_split_at(split_index, unprocessed_floats)
    decoded_value = attribute.decode(encoded_values, attr)
    next_ks_vs = utilities \
        .lazy_cons((attr.key, decoded_value), processed_ks_vs)
    return next_floats, next_ks_vs


# public
def decode(encoded_datapoint: Sequence,
           preprocessor: Preprocessor
           ) -> Dict[str, str]:
    return preprocessor.fold_left(
        (encoded_datapoint, seq([])),
        lambda acc, x: decode_acc_f(acc, x))[1] \
        .to_dict()


# public
def to_json(preprocessor: Preprocessor) -> str:
    return json.dumps(
        preprocessor \
            .map(lambda x: attribute_serialized.serialized(x)) \
            .to_list(),
        separators=(',', ':'),
        cls=utilities.EnhancedJSONEncoder
    )


# public
def of_json(s: str) -> Preprocessor:
    return seq(json.loads(s)) \
        .map(lambda x: attribute_serialized.deserialized(x))
