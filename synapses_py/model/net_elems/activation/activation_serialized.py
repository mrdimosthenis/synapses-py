from synapses_py.model.net_elems.activation import activation

ActivationSerialized = str


def serialized(activ_f: activation.Activation) -> ActivationSerialized:
    return activ_f.name


def deserialized(s: ActivationSerialized) -> activation.Activation:
    return {
        'sigmoid': activation.sigmoid,
        'identity': activation.identity,
        'tanh': activation.tanh,
        'leakyReLU': activation.leakyReLU,
    }[s]
