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
import logging

def suppress_wandb_log_messages():
    """ Patches wandb logger to suppress upload messages, commonly occuring on KeyboardInterrupt
        or program crash.

        To print the log url:
        run = wandb.init(...)
        print(run.url)
    """

    # (1) kill off all wandb logger output below CRITICAL
    logging.getLogger("wandb").setLevel(logging.CRITICAL)

    # (2) monkey‐patch any of the internal "_footer…" functions to no‐ops
    def _suppress_footer(*args, **kwargs):
        return None

    # Depending on your wandb version these lives under sdk.internal.file_pusher
    try:
        import wandb.sdk.internal.file_pusher as _fp
        for name in dir(_fp):
            if name.startswith("_footer"):
                setattr(_fp, name, _suppress_footer)
    except ImportError:
        pass

    # There is also a per‐run footer in
    # wandb.sdk.internal.run._footer_single_run_status_info
    try:
        import wandb.sdk.internal.run as _run
        if hasattr(_run, "_footer_single_run_status_info"):
            _run._footer_single_run_status_info = _suppress_footer
    except ImportError:
        pass
