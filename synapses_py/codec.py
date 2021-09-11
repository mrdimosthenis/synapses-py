from typing import List, Optional, Iterable, Dict, Tuple

from functional import seq

from synapses_py.model.encoding import preprocessor


class Codec:

    def __init__(self,
                 attributes: Optional[List[Tuple[str, bool]]] = None,
                 data_points: Optional[Iterable[Dict[str, str]]] = None,
                 json: Optional[str] = None):
        if json is not None:
            self.__contents = preprocessor.of_json(json)
        else:
            keys_with_flags = seq(attributes)
            dataset = seq(data_points)
            self.__contents = preprocessor.init(keys_with_flags, dataset)

    def encode(self, data_point: Dict[str, str]) -> List[float]:
        return preprocessor \
            .encode(data_point, self.__contents) \
            .to_list()

    def decode(self, encoded_values: List[float]) -> Dict[str, str]:
        encoded_datapoint = seq(encoded_values)
        return preprocessor \
            .decode(encoded_datapoint, self.__contents)

    def json(self) -> str:
        return preprocessor.to_json(self.__contents)
