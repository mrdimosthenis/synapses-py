from typing import List, Dict

from dataclasses import dataclass
from functional import seq

from synapses_py.model.encoding.attribute import attribute


@dataclass(frozen=True)
class AttributeSerialized:
    key: str


@dataclass(frozen=True)
class DiscreteAttributeSerialized(AttributeSerialized):
    key: str
    values: List[str]


@dataclass(frozen=True)
class ContinuousAttributeSerialized(AttributeSerialized):
    key: str
    min: float
    max: float


def serialized(attr: attribute.Attribute) -> Dict:
    if isinstance(attr, attribute.DiscreteAttribute):
        return {
            "Case": "SerializableDiscrete",
            "Fields": [
                DiscreteAttributeSerialized(
                    attr.key,
                    attr.values.to_list()
                )
            ]
        }
    elif isinstance(attr, attribute.ContinuousAttribute):
        return {
            "Case": "SerializableContinuous",
            "Fields": [
                ContinuousAttributeSerialized(
                    attr.key,
                    attr.min,
                    attr.max
                )
            ]
        }
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')


def deserialized(dictionary: Dict) -> attribute.Attribute:
    case = dictionary.get("Case")
    fields = dictionary['Fields'][0]
    if case == "SerializableDiscrete":
        return attribute.DiscreteAttribute(
            fields['key'],
            seq(fields['values'])
        )
    elif case == "SerializableContinuous":
        return attribute.ContinuousAttribute(
            fields['key'],
            fields['min'],
            fields['max']
        )
    else:
        raise Exception('Attribute is neither Discrete nor Continuous')
