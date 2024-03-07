import importlib
import inspect
import random
from copy import deepcopy
from typing import Any, Callable, Dict, Type

import numpy as np
import torch


def load_module(name):
    if ":" in name:
        mod_name, attr_name = name.split(":")
    else:
        li = name.split(".")
        mod_name, attr_name = ".".join(li[:-1]), li[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_object(init_class: Type[object], possible_args: Dict[str, Any], **kwargs):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(init_class.__init__).args
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    new_object = init_class(**params)
    return new_object


def run_method(
    method: Callable,
    possible_args: Dict[str, Any],
    **kwargs,
):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(method).args
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    method(**params)
