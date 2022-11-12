
import sys
import gravlearn

# TODO (ashutiwa): remove all print statemenst with logging
def set_snakemake_config(param, value, field_name="snakemake.params"):
    if field_name not in _CONFIG_STORE:
        _CONFIG_STORE[field_name] = {}
    _CONFIG_STORE[field_name][param] = value


def _get_sk_value(param, sk_field, default=None):
    val = default
    try:
        val = sk_field[param]
    except:
        val = _CONFIG_STORE.get(sk_field, {}).get(param, default)
    # TODO (ashutiwa): remove this print statement
    assert val is not None
    return val

def _wrap_param(val, object=None, type=None):
    assert not (object and type), "Cannot specify both object and type"
    if object:
        return _OBJECT_MAP.get(val, val)
    if type:
        return type(val)
    return val


def get_sk_value(param, field, default=None, object=False, type=None):
    val = _wrap_param(_get_sk_value(param, sk_field=field, default=default), object=object, type=type)
    print("returning value: ", val, " for param: ", param)
    return val


def get_input_value(param_idx, default=None, type=None):
    val = default
    try:
        val = sys.argv[param_idx]
    except IndexError:
        pass
    if type is not None:
        return type(val)
    return val


class CONSTANTS(object):
    class DIST_METRIC(object):
        DOTSIM = 'dotsim'
    DEVICE = "device"
    PARAMS = "params"
    OUTPUT = "output"


_OBJECT_MAP = {
    CONSTANTS.DIST_METRIC.DOTSIM: gravlearn.metrics.DistanceMetrics.DOTSIM
}

_CONFIG_STORE = {
    # "params": {
    #     CONSTANTS.DIST_METRIC.__name__.lower(): CONSTANTS.DIST_METRIC.DOTSIM,
    #     "checkpoint": 10,
    #     "lr": .001,
    #     "dim": 100,},
    # "output": {"outfile": "fairness_model_ck.pt"}
}