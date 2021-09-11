from typing import Dict

from dataclasses import dataclass
from functional import seq
from functional.pipeline import Sequence

from synapses_py.model import utilities


@dataclass(frozen=True)
class Attribute:
    key: str


@dataclass(frozen=True)
class DiscreteAttribute(Attribute):
    key: str
    values: Sequence


@dataclass(frozen=True)
class ContinuousAttribute(Attribute):
    key: str
    min: float
    max: float


def realize(attr: Attribute) -> Attribute:
    if isinstance(attr, DiscreteAttribute):
        return DiscreteAttribute(attr.key, attr.values.cache())
    elif isinstance(attr, ContinuousAttribute):
        return attr
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')


def parse(s: str) -> float:
    return float(s.strip())


def updated(datapoint: Dict[str, str], attr: Attribute) -> Attribute:
    if isinstance(attr, DiscreteAttribute):
        exist = attr \
            .values \
            .exists(lambda x: x == datapoint[attr.key])
        updated_vals = attr.values if exist else \
            utilities.lazy_cons(
                datapoint[attr.key],
                attr.values
            )
        return DiscreteAttribute(attr.key, updated_vals)
    elif isinstance(attr, ContinuousAttribute):
        v = parse(datapoint[attr.key])
        return ContinuousAttribute(
            attr.key,
            min(v, attr.min),
            max(v, attr.max)
        )
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')


def encode(value: str, attr: Attribute) -> Sequence:
    if isinstance(attr, DiscreteAttribute):
        return attr \
            .values \
            .map(lambda x: 1.0 if x == value else 0.0)
    elif isinstance(attr, ContinuousAttribute):
        return seq([0.5]) if \
            (attr.min == attr.max) else \
            seq([(parse(value) - attr.min) /
                 (attr.max - attr.min)])
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')


def decode(encoded_values: Sequence,
           attr: Attribute
           ) -> str:
    if isinstance(attr, DiscreteAttribute):
        return attr \
            .values \
            .zip(encoded_values) \
            .reduce(lambda acc, x:
                    x if x[1] > acc[1] else
                    acc)[0]
    elif isinstance(attr, ContinuousAttribute):
        if attr.min == attr.max:
            v = attr.min
        else:
            v = encoded_values.head() * \
                (attr.max - attr.min) + \
                attr.min
        return str(v)
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')
