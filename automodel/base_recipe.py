import torch.nn as nn
from torch.optim import Optimizer

def has_load_restore_state(object):
    return all(
        callable(getattr(object, attr, None))
        for attr in ('load_state_dict', 'state_dict')
    )

class BaseRecipe:
    """
    Checkpoint registry
    """
    def __setattr__(self, key, value):
        """ Overriden __setattr__ to keep track of stateful classes.

        Args:
            key (str): attribute named.
            value (Any): Value assigned

        Raises:
            ValueError: if __state_tracked is attemped to be overwriten.

        """
        # assuming no one will do recipe.__dict__['__state_tracked'] = None
        if key == '__state_tracked':
            raise ValueError("cannot set __state_tracked")
        if not '__state_tracked' in self.__dict__:
            self.__dict__['__state_tracked'] = set()
        if isinstance(value, (nn.Module, Optimizer)) or has_load_restore_state(value):
            assert not key in self.__dict__['__state_tracked']
            self.__dict__['__state_tracked'].add(key)
        super().__setattr__(key, value)
