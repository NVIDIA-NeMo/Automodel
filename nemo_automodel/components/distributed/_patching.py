# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Access to attributes that an object's declared type does not include.

The distributed strategies patch objects owned by external libraries that expose no
extension API for it: FSDP2 and pipelining internals, ``torch.nn.functional``,
Hugging Face attention modules and Megatron-FSDP parameters. They replace functions and
methods, set marker attributes on the patched callables, and attach per-object state.
None of these attributes are declared by the objects' types (``nn.Module.__setattr__``
even narrows assigned values to ``Tensor | Module``), so a type checker rejects writing
them directly.

Every such access in ``nemo_automodel.components.distributed`` goes through the two
functions below, which keeps the package's reflection in one documented place, as
CODING_GUIDELINES.md ("Avoid reflection") requires. Callers validate the type of values
they read back.
"""


def set_undeclared_attr(owner: object, name: str, value: object) -> None:
    """Set ``owner.<name>`` to ``value`` where ``owner``'s declared type lacks that attribute.

    Assignment goes through the owner's normal ``__setattr__`` (for example
    ``nn.Module.__setattr__``), exactly as a direct ``owner.<name> = value`` would.

    Args:
        owner: The external object (module, class, function, or instance) being patched.
        name: The attribute to set.
        value: The value to install.
    """
    setattr(owner, name, value)


def get_undeclared_attr(owner: object, name: str) -> object:
    """Return ``owner.<name>``, or ``None`` when the attribute is absent.

    Args:
        owner: The external object whose undeclared attribute is read.
        name: The attribute to read.

    Returns:
        The attribute's value, unvalidated; callers narrow it before use.
    """
    return getattr(owner, name, None)
