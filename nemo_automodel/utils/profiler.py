# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""
Profiling utilities for training performance analysis.

This module provides basic profiling capabilities for PyTorch training,
controlling when to start and stop profiling for kernel trace collection.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Union
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class ProfilerConfig:
    """Configuration for the profiler.
    
    Args:
        enabled: Whether profiling is enabled.
        profile_step: Step number to start profiling (1-indexed).
        profile_duration: Number of steps to profile (0 = until training ends).
        output_dir: Directory to save profiling outputs.
        cuda_profiler_enabled: Whether to use CUDA profiler APIs.
        verbose: Whether to log profiling actions.
    """
    enabled: bool = False
    profile_step: int = 30
    profile_duration: int = 1
    output_dir: str = "profiling_outputs"
    cuda_profiler_enabled: bool = True
    verbose: bool = True


class NsysProfiler:
    """
    NSYS profiler for training performance analysis.
    
    Provides simple start/stop control for profiling training steps
    with configurable start/stop conditions.
    
    Example:
        ```python
        profiler = NsysProfiler(config)
        
        for step, batch in enumerate(dataloader):
            profiler.step_begin(step + 1)
            
            # Your training code here
            outputs = model(batch)
            loss.backward()
            optimizer.step()
                
            if profiler.step_end():
                break  # Stop training after profiling
        ```
    """

    def __init__(self, config: Optional[Union[ProfilerConfig, dict]] = None):
        """Initialize the profiler.
        
        Args:
            config: Profiler configuration or dict to create ProfilerConfig.
        """
        if config is None:
            config = ProfilerConfig()
        elif isinstance(config, dict):
            config = ProfilerConfig(**config)
            
        self.config = config
        self.current_step = 0
        self.profiling_started = False
        self.profile_step_count = 0
            
        if config.enabled and config.verbose:
            logger.info(f"Profiler initialized: will profile at step {config.profile_step} "
                       f"for {config.profile_duration} steps")

    def step_begin(self, step: int) -> bool:
        """
        Call at the beginning of each training step.
        
        Args:
            step: Current training step (1-indexed).
            
        Returns:
            bool: True if profiling is active for this step.
        """
        self.current_step = step
        
        if not self.config.enabled:
            return False
            
        # Check if we should start profiling
        if (step == self.config.profile_step and not self.profiling_started):
            self._start_profiling()
            
        return self.profiling_started

    def step_end(self) -> bool:
        """
        Call at the end of each training step.
        
        Returns:
            bool: True if training should stop (profiling complete).
        """
        if not self.profiling_started:
            return False
            
        self.profile_step_count += 1
        
        # Check if we should stop profiling
        if (self.config.profile_duration > 0 and 
            self.profile_step_count >= self.config.profile_duration):
            self._stop_profiling()
            return True
            
        return False

    def _start_profiling(self):
        """Start profiling with CUDA profiler."""
        self.profiling_started = True
        
        if self.config.verbose:
            logger.info(f"🚀 Starting profiler at step {self.current_step}")
            
        if self.config.cuda_profiler_enabled:
            try:
                torch.cuda.cudart().cudaProfilerStart()
            except Exception as e:
                logger.warning(f"Failed to start CUDA profiler: {e}")

    def _stop_profiling(self):
        """Stop profiling and cleanup."""
        if self.config.cuda_profiler_enabled:
            try:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
            except Exception as e:
                logger.warning(f"Failed to stop CUDA profiler: {e}")
                
        if self.config.verbose:
            logger.info("✅ Profiler stopped. Check nsys output for results.")
            
        # Brief pause to ensure profiling data is flushed
        time.sleep(1)

    def is_profiling(self) -> bool:
        """Check if profiling is currently active."""
        return self.profiling_started


def create_profiler_from_config(cfg) -> NsysProfiler:
    """
    Create a profiler from configuration.
    
    Args:
        cfg: Configuration object with profiler settings.
        
    Returns:
        NsysProfiler instance.
    """
    profiler_cfg = cfg.get("profiler", {})
    
    # Convert from config format to ProfilerConfig
    config = ProfilerConfig(
        enabled=profiler_cfg.get("enabled", False),
        profile_step=profiler_cfg.get("profile_step", 30),
        profile_duration=profiler_cfg.get("profile_duration", 1),
        output_dir=profiler_cfg.get("output_dir", "profiling_outputs"),
        cuda_profiler_enabled=profiler_cfg.get("cuda_profiler_enabled", True),
        verbose=profiler_cfg.get("verbose", True),
    )
    
    return NsysProfiler(config)


 