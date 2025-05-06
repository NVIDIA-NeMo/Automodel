import logging
import os
import signal
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(kw_only=True)
class RNGConfig:
    """Configuration settings for random number generation."""

    seed: int = 1234
    """Random seed used for python, numpy, pytorch, and cuda."""

    te_rng_tracker: bool = False
    """Use the Transformer Engine version of the random number generator.
    Required for CUDA graphs support."""

    inference_rng_tracker: bool = False
    """Use a random number generator configured for inference."""

    data_parallel_random_init: bool = False
    """Enable random initialization of params across data parallel ranks"""


@dataclass(kw_only=True)
class TrainingConfig:
    """Configuration settings related to the training loop and validation."""

    # ---------------- Training config. ----------------

    micro_batch_size: Optional[int] = None
    """Batch size per model instance (local batch size). Global batch size is local batch size times
    data parallel size times number of micro batches."""

    global_batch_size: Optional[int] = None
    """Training batch size. If set, it should be a multiple of micro-batch-size times
    data-parallel-size. If this value is None, then use micro-batch-size * data-parallel-size
    as the global batch size. This choice will result in 1 for number of micro-batches."""

    rampup_batch_size: Optional[list[int]] = None
    """Batch size ramp up with the following values: <start batch size>, <batch size increment>,
    <ramp-up samples>
    For example:
        rampup-batch-size = [16, 8, 300000]
        global-batch-size 1024
    will start with global batch size 16 and over (1024 - 16) / 8 = 126 intervals will increase
    the batch size linearly to 1024. In each interval we will use approximately
    300000 / 126 = 2380 samples.
    """

    decrease_batch_size_if_needed: bool = False
    """If set, decrease batch size if microbatch_size * dp_size does not divide batch_size.
    Useful for KSO (Keep Soldiering On) to continue making progress if number of healthy GPUs
    (and corresponding dp_size) does not support current batch_size. Old batch_size will be
    restored if training is re-started with dp_size that divides batch_size // microbatch_size."""

    empty_unused_memory_level: Literal[0, 1, 2] = 0
    """Call torch.cuda.empty_cache() each iteration (training and eval), to reduce fragmentation.
    0=off, 1=moderate, 2=aggressive.
    """

    check_weight_hash_across_dp_replicas_interval: Optional[int] = None
    """Interval to check weight hashes are same across DP replicas. If not specified, weight hashes not checked."""

    train_sync_interval: Optional[int] = None
    """Training CPU-GPU synchronization interval, to ensure that CPU is not running too far ahead of GPU."""

    train_iters: Optional[int] = None
    """Total number of iterations to train over all training runs.
    Note that either train-iters or train-samples should be provided.
    """

    exit_interval: Optional[int] = None
    """Exit the program after the iteration is divisible by this value."""

    exit_duration_in_mins: Optional[int] = None
    """Exit the program after this many minutes."""

    exit_signal_handler: bool = False
    """Dynamically save the checkpoint and shutdown the training if SIGTERM is received"""

    exit_signal: int = signal.SIGTERM
    """Signal for the signal handler to detect."""

    exit_signal_handler_for_dataloader: bool = False
    """Use signal handler for dataloader workers"""

    manual_gc: bool = False
    """Disable the threshold-based default garbage collector and trigger the garbage collection
    manually. Manual garbage collection helps to align the timing of the collection across ranks
    which mitigates the impact of CPU-associated jitters. When the manual gc is enabled, garbage
    collection is performed only at the start and the end of the validation routine by default."""

    manual_gc_interval: int = 0
    """Training step interval to trigger manual garbage collection.
    When the value is set to 0, garbage collection is not triggered between training steps.
    """

    manual_gc_eval: bool = True
    """When using manual garbage collection,
    disable garbage collection at the start and the end of each evaluation run.
    """

    # ---------------- Validation config. ----------------

    eval_iters: int = 100
    """Number of iterations to run for evaluation validation/test for."""

    eval_interval: Optional[int] = 1000
    """Interval between running evaluation on validation set."""

    skip_train: bool = False
    """If set, bypass the training loop, optionally do evaluation for validation/test, and exit."""


@dataclass(kw_only=True)
class DistributedInitConfig:
    """Configuration settings for distributed training initialization."""

    # ---------------- Distributed config. ----------------

    distributed_backend: Literal["nccl", "gloo"] = "nccl"
    """Which backend to use for distributed training."""

    distributed_timeout_minutes: int = 10
    """Timeout minutes for torch.distributed."""

    align_grad_reduce: bool = True
    """If not set, all PP stages will launch gradient reduces simultaneously.
    Otherwise, each PP stage will independently launch as needed.
    """

    local_rank: int = field(default_factory=lambda: int(os.getenv("LOCAL_RANK", "0")))
    """local rank passed from distributed launcher."""

    lazy_init: bool = False
    """If set to True, initialize_megatron() skips DDP initialization and returns function to complete it instead. Also turns on --use-cpu-initialization flag. This is for external DDP manager."""

    use_torch_fsdp2: bool = False
    """Use the torch FSDP2 implementation. FSDP2 is not currently working with Pipeline Parallel.
    It is still not in a stable release stage, and may therefore contain bugs or other
    potential issues."""

    nccl_communicator_config_path: Optional[str] = None
    """Path to the yaml file with NCCL communicator configurations. The number of min/max thread
    groups and thread group cluster size of each communicator can be configured by setting
    `min_ctas`, `max_ctas`, and `cga_cluster_size`."""

    use_tp_pp_dp_mapping: bool = False
    """If set, distributed ranks initialize order is changed from tp-dp-pp to tp-pp-dp.
    Make sure EP and CP aren't used with this option enabled.
    """

    use_gloo_process_groups: bool = True
    """If set, create Gloo process groups for communications."""


@dataclass(kw_only=True)
class ProfilingConfig:
    """Configuration settings for profiling the training process."""

    # ---------------- Profiling config. ----------------

    use_nsys_profiler: bool = False
    """Enable nsys profiling. When using this option, nsys options should be specified in
    commandline. An example nsys commandline is
    `nsys profile -s none -t nvtx,cuda -o <path/to/output_file> --force-overwrite true
    --capture-range=cudaProfilerApi --capture-range-end=stop`.
    """

    profile_step_start: int = 10
    """Global step to start profiling."""

    profile_step_end: int = 12
    """Global step to stop profiling."""

    use_pytorch_profiler: bool = False
    """Use the built-in pytorch profiler. Useful if you wish to view profiles in tensorboard."""

    profile_ranks: list[int] = field(default_factory=lambda: [0])
    """Global ranks to profile."""

    record_memory_history: bool = False
    """Record memory history in last rank."""

    memory_snapshot_path: str = "snapshot.pickle"
    """Specifies where to dump the memory history pickle."""

    record_shapes: bool = False
    """Record shapes of tensors."""


@dataclass(kw_only=True)
class LoggerConfig:
    """Configuration settings for logging, including TensorBoard and WandB."""

    # ---------------- Logging config. ----------------

    log_interval: int = 100
    """Report loss and timing interval."""

    log_params_norm: bool = False
    """If set, calculate and log parameters norm."""

    log_throughput: bool = False
    """If set, calculate and log throughput per GPU."""

    log_progress: bool = False
    """If set, log progress (in terms of number of processed tokens and number of floating-point operations)
    to progress.txt file in checkpoint directory.
    """

    timing_log_level: Literal[0, 1, 2] = 0
    """Granularity level to measure and report timing.
    0: report only iteration time and make sure timing does not introduce extra overhead.
    1: report timing for operations that are executed very limited times (basically once) during each iteration
        (such as gradient all-reduce)
    2: report timing for operations that migh be executed numerous times during each iteration.
    Note that setting the level to 1 or 2 might cause increase in iteration time.
    """

    timing_log_option: Literal["max", "minmax", "all"] = "minmax"
    """Options for logging timing:
    max: report the max timing across all ranks
    minmax: report min and max timings across all ranks
    all: report timings of all ranks.
    """

    tensorboard_dir: Optional[str] = None
    """Write TensorBoard logs to this directory."""

    tensorboard_log_interval: int = 1
    """Report to tensorboard interval."""

    tensorboard_queue_size: int = 1000
    """Size of the tensorboard queue for pending events and summaries
    before one of the 'add' calls forces a flush to disk.
    """

    log_timers_to_tensorboard: bool = False
    """If set, write timers to tensorboard."""

    log_loss_scale_to_tensorboard: bool = True
    """Disable loss-scale logging to tensorboard."""

    log_validation_ppl_to_tensorboard: bool = False
    """If set, write validation perplexity to tensorboard."""

    log_memory_to_tensorboard: bool = False
    """Enable memory logging to tensorboard."""

    log_world_size_to_tensorboard: bool = False
    """Enable world size logging to tensorboard."""

    wandb_project: Optional[str] = None
    """The wandb project name. Ignore wandb by default."""

    wandb_exp_name: Optional[str] = None
    """The wandb experiment name."""

    wandb_save_dir: Optional[str] = None
    """Path to save the wandb results locally."""

    wandb_entity: Optional[str] = None
    """The wandb entity name."""

    logging_level: int = logging.INFO
    """Set default logging level"""

    filter_warnings: bool = True
    """Filter out warning messages"""

    modules_to_filter: Optional[list[str]] = None
    """List of modules to filter out from the logs"""

    set_level_for_all_loggers: bool = False
    """Set the logging level for all loggers. If False, only level for NeMo loggers will be set."""
