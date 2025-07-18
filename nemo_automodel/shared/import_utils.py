# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file is taken from https://github.com/NVIDIA/NeMo-Curator, which is adapted from cuML's safe_imports module:
# https://github.com/rapidsai/cuml/blob/e93166ea0dddfa8ef2f68c6335012af4420bc8ac/python/cuml/internals/safe_imports.py


import importlib
import logging
import traceback
from contextlib import contextmanager

import torch
from packaging.version import Version as PkgVersion

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

GPU_INSTALL_STRING = (
    """Install GPU packages via `pip install --extra-index-url """
    """https://pypi.nvidia.com nemo-curator[cuda12x]`
or use `pip install --extra-index-url https://pypi.nvidia.com ".[cuda12x]"` if installing from source"""
)
MISSING_TRITON_MSG = "triton is not installed. Please install it with `pip install triton`."
MISSING_QWEN_VL_UTILS_MSG = "qwen_vl_utils is not installed. Please install it with `pip install qwen-vl-utils`."
MISSING_CUT_CROSS_ENTROPY_MSG = (
    "cut_cross_entropy is not installed. Please install it with `pip install cut-cross-entropy`."
)


class UnavailableError(Exception):
    """
    Error thrown if a symbol is unavailable due to an issue importing it.
    """


@contextmanager
def null_decorator(*args, **kwargs):
    """
    No-op decorator.
    """
    if len(kwargs) == 0 and len(args) == 1 and callable(args[0]):
        return args[0]
    else:

        def inner(func):
            return func

        return inner


class UnavailableMeta(type):  # noqa D105
    """
    A metaclass for generating placeholder objects for unavailable symbols.

    This metaclass allows errors to be deferred from import time to the time
    that a symbol is actually used in order to streamline the usage of optional
    dependencies. This is particularly useful for attempted imports of GPU-only
    modules which will only be invoked if GPU-only functionality is
    specifically used.

    If an attempt to import a symbol fails, this metaclass is used to generate
    a class which stands in for that symbol. Any attempt to call the symbol
    (instantiate the class) or access its attributes will throw an
    UnavailableError exception. Furthermore, this class can be used in
    e.g. isinstance checks, since it will (correctly) fail to match any
    instance it is compared against.

    In addition to calls and attribute access, a number of dunder methods are
    implemented so that other common usages of imported symbols (e.g.
    arithmetic) throw an UnavailableError, but this is not guaranteed for
    all possible uses. In such cases, other exception types (typically
    TypeErrors) will be thrown instead.
    """

    def __new__(meta, name, bases, dct):  # noqa D105
        if dct.get("_msg", None) is None:
            dct["_msg"] = f"{name} could not be imported"
        name = f"MISSING{name}"
        return super(UnavailableMeta, meta).__new__(meta, name, bases, dct)

    def __call__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __getattr__(cls, name):  # noqa D105
        raise UnavailableError(cls._msg)

    def __eq__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __lt__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __gt__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __le__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ge__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ne__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __abs__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __add__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __radd__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __iadd__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __floordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rfloordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ifloordiv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __lshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rlshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __mul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rmul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __imul__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ilshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __pow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rpow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __ipow__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rrshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __irshift__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __sub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rsub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __isub__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __truediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rtruediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __itruediv__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __divmod__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __rdivmod__(cls, other):  # noqa D105
        raise UnavailableError(cls._msg)

    def __neg__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __invert__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __hash__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __index__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __iter__(cls):  # noqa D105
        raise UnavailableError(cls._msg)

    def __delitem__(cls, name):  # noqa D105
        raise UnavailableError(cls._msg)

    def __setitem__(cls, name, value):  # noqa D105
        raise UnavailableError(cls._msg)

    def __enter__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __get__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __delete__(cls, *args, **kwargs):  # noqa D105
        raise UnavailableError(cls._msg)

    def __len__(cls):  # noqa D105
        raise UnavailableError(cls._msg)


def is_unavailable(obj):
    """
    Helper to check if given symbol is actually a placeholder.
    """
    return type(obj) is UnavailableMeta


def safe_import(module, *, msg=None, alt=None):
    """
    A function used to import modules that may not be available.

    This function will attempt to import a module with the given name, but it
    will not throw an ImportError if the module is not found. Instead, it will
    return a placeholder object which will raise an exception only if used.

    Args:
        module (str): The name of the module to import.
        msg (str or None): An optional error message to be displayed if this module is used
            after a failed import.
        alt (object): An optional module to be used in place of the given module if it
            fails to import

    Returns:
        Tuple(bool, object): The imported module, the given alternate, or a class derived from
        UnavailableMeta, and a boolean indicating whether the intended import was successful.
    """
    try:
        return True, importlib.import_module(module)
    except ImportError:
        exception_text = traceback.format_exc()
        logger.debug(f"Import of {module} failed with: {exception_text}")
    except Exception:
        exception_text = traceback.format_exc()
        raise
    if msg is None:
        msg = f"{module} could not be imported"
    if alt is None:
        return False, UnavailableMeta(module.rsplit(".")[-1], (), {"_msg": msg})
    else:
        return False, alt


def safe_import_from(module, symbol, *, msg=None, alt=None, fallback_module=None):
    """
    A function used to import symbols from modules that may not be available.

    This function will attempt to import a symbol with the given name from
    the given module, but it will not throw an ImportError if the symbol is not
    found. Instead, it will return a placeholder object which will raise an
    exception only if used.

    Args:
        module (str): The name of the module in which the symbol is defined.
        symbol (str): The name of the symbol to import.
        msg (str or None): An optional error message to be displayed if this symbol is used
            after a failed import.
        alt (object): An optional object to be used in place of the given symbol if it fails
            to import
        fallback_module (str): Alternative name of the model in which the symbol is defined.
            The function will first to import using the `module` value and if that fails will also
            try the `fallback_module`.

    Returns:
        Tuple(object, bool): The imported symbol, the given alternate, or a class derived from
        UnavailableMeta, and a boolean indicating whether the intended import was successful.
    """
    try:
        imported_module = importlib.import_module(module)
        return True, getattr(imported_module, symbol)
    except ImportError:
        exception_text = traceback.format_exc()
        logger.debug(f"Import of {module} failed with: {exception_text}")
    except AttributeError:
        # if there is a fallback module try it.
        if fallback_module is not None:
            return safe_import_from(fallback_module, symbol, msg=msg, alt=alt, fallback_module=None)
        exception_text = traceback.format_exc()
        logger.info(f"Import of {symbol} from {module} failed with: {exception_text}")
    except Exception:
        exception_text = traceback.format_exc()
        raise
    if msg is None:
        msg = f"{module}.{symbol} could not be imported"
    if alt is None:
        return False, UnavailableMeta(symbol, (), {"_msg": msg})
    else:
        return False, alt


def gpu_only_import(module, *, alt=None):
    """
    A function used to import modules required only in GPU installs.

    This function will attempt to import a module with the given name.
    This function will attempt to import a symbol with the given name from
    the given module, but it will not throw an ImportError if the symbol is not
    found. Instead, it will return a placeholder object which will raise an
    exception only if used with instructions on installing a GPU build.

    Args:
        module (str): The name of the module to import.
        alt (object): An optional module to be used in place of the given module if it
            fails to import in a non-GPU-enabled install

    Returns:
        object: The imported module, the given alternate, or a class derived from
        UnavailableMeta.
    """
    return safe_import(
        module,
        msg=f"{module} is not enabled in non GPU-enabled installations or environemnts. {GPU_INSTALL_STRING}",
        alt=alt,
    )


def gpu_only_import_from(module, symbol, *, alt=None):
    """
    A function used to import symbols required only in GPU installs.

    This function will attempt to import a module with the given name.
    This function will attempt to import a symbol with the given name from
    the given module, but it will not throw an ImportError if the symbol is not
    found. Instead, it will return a placeholder object which will raise an
    exception only if used with instructions on installing a GPU build.

    Args:
        module (str): The name of the module to import.
        symbol (str): The name of the symbol to import.
        alt (object): An optional object to be used in place of the given symbol if it fails
            to import in a non-GPU-enabled install

    Returns:
        object: The imported symbol, the given alternate, or a class derived from
        UnavailableMeta.
    """
    return safe_import_from(
        module,
        symbol,
        msg=f"{module}.{symbol} is not enabled in non GPU-enabled installations or environments. {GPU_INSTALL_STRING}",
        alt=alt,
    )


def get_torch_version():
    """
    Return pytorch version with fallback if unavailable.

    Returns:
        PkgVersion: Pytorch's version
    """
    try:
        _torch_version = PkgVersion(torch.__version__)
    except Exception:
        # This is a WAR for building docs, where torch is not actually imported
        _torch_version = PkgVersion("0.0.0")
    return _torch_version


def is_torch_min_version(version, check_equality=True):
    """
    Check if minimum version of `torch` is installed.
    """
    if check_equality:
        return get_torch_version() >= PkgVersion(version)
    return get_torch_version() > PkgVersion(version)
