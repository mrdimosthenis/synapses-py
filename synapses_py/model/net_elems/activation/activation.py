import math
from typing import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Activation:
    name: str
    f: Callable[[float], float]
    deriv: Callable[[float], float]
    inverse: Callable[[float], float]


def sigmoid_f(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


sigmoid: Activation = Activation(
    'sigmoid',
    sigmoid_f,
    lambda d: sigmoid_f(d) * (1.0 - sigmoid_f(d)),
    lambda y: math.log(y / (1.0 - y))
)

identity: Activation = Activation(
    'identity',
    lambda x: x,
    lambda _: 1.0,
    lambda y: y
)

tanh: Activation = Activation(
    'tanh',
    math.tanh,
    lambda d: 1.0 - math.tanh(d) * math.tanh(d),
    lambda y: 0.5 * math.log((1.0 + y) / (1.0 - y))
)

leakyReLU: Activation = Activation(
    'leakyReLU',
    lambda x: 0.01 * x if x < 0.0 else x,
    lambda d: 0.01 if d < 0.0 else 1.0,
    lambda y: y / 0.01 if y < 0.0 else y
)
