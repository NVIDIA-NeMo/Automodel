
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from transformers import AutoModelForCausalLM

from nemo_automodel.shared.import_utils import safe_import
HAS_LIGER_KERNEL, liger_kernel_trf = safe_import('liger_kernel.transformers')


class NeMoAutoModelForCausalLM(AutoModelForCausalLM):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        use_liger_kernel = kwargs.pop('use_liger_kernel', True)
        ans = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return ans
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=ans)
            except Exception as e:
                del ans
                # If patching failed, retry
                return cls.from_pretrained(
                    pretrained_model_name_or_path, *model_args, **kwargs, use_liger_kernel=False)
        return ans


    @classmethod
    def from_config(cls, config, **kwargs):
        use_liger_kernel = kwargs.pop('use_liger_kernel', True)
        ans = super().from_config(config, **kwargs)
        if use_liger_kernel:
            if not HAS_LIGER_KERNEL:
                logging.warning("Asked to use Liger Kernel, but could not import")
                return ans
            try:
                liger_kernel_trf._apply_liger_kernel_to_instance(model=ans)
            except Exception as e:
                del ans
                # If patching failed, retry
                return cls.from_config(config, **kwargs, use_liger_kernel=False)
        return ans
