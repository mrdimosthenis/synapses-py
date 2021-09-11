from typing import Iterable, Tuple, List

from functional import seq

from synapses_py.model import mathematics


def rmse(output_pairs: Iterable[Tuple[List[float], List[float]]]) -> float:
    y_hats_with_ys = seq(output_pairs) \
        .map(lambda t: (seq(t[0]), seq(t[1])))
    return mathematics.root_mean_square_error(y_hats_with_ys)


def score(output_pairs: Iterable[Tuple[List[float], List[float]]]) -> float:
    y_hats_with_ys = seq(output_pairs) \
        .map(lambda t: (seq(t[0]), seq(t[1])))
    return mathematics.accuracy(y_hats_with_ys)
