# Parallelism in NeMo AutoModel

NeMo AutoModel supports various data-parallel and model-parallel deep learning workload deployment methods, which can be mixed together arbitrarily.

## Data Parallelism

Data Parallelism (DP) replicates the model across multiple GPUs. Data batches are evenly distributed between GPUs and the data-parallel GPUs process them independently. While the computation workload is efficiently distributed across GPUs, inter-GPU communication is required to keep the model replicas consistent between training steps.

### Distributed Data Parallelism

Distributed Data Parallelism (DDP) keeps model copies consistent by synchronizing parameter gradients across data-parallel GPUs before each parameter update. It sums gradients of all model copies using all-reduce communication collectives.

### Distributed Optimizer

The distributed optimizer is a memory-optimized data-parallel method that shards optimizer states and high-precision master parameters across GPUs instead of replicating them. It uses reduce-scatter for gradients and all-gather for parameters, reducing memory requirements for large-scale training.

#### Enable Data Parallelism

In NeMo AutoModel, DDP is the default parallel deployment method. The total number of GPUs corresponds to the size of the DP group.

### Fully Sharded Data Parallel (FSDP2)

FSDP2 is an advanced data parallelism technique that shards model parameters, gradients, and optimizer states across all GPUs. It offers:

- Memory efficiency by only keeping needed shards on each GPU
- Flexible precision control for parameters and gradients
- Overlapping computation and communication

#### Enable FSDP2

To enable FSDP2 in NeMo AutoModel:

1. Set the strategy to FSDP2 in your training configuration
2. Configure sharding strategy (FULL_SHARD, SHARD_GRAD_OP, etc.)
3. Set mixed precision policies
4. Configure CPU offload if needed

Example configuration options:
- `sharding_strategy`: FULL_SHARD, SHARD_GRAD_OP, or NO_SHARD
- `mixed_precision`: Policy for parameter, buffer, and reduction precision
- `cpu_offload`: Offload parameters and gradients to CPU
- `backward_prefetch`: Control backward prefetching strategy

FSDP2 is particularly effective for very large models where memory constraints are critical.

## Model Parallelism

Model Parallelism (MP) partitions model parameters across GPUs to reduce per-GPU memory requirements. NeMo AutoModel supports various model-parallel methods.

### Tensor Parallelism

Tensor Parallelism (TP) distributes parameter tensors of individual layers across GPUs. This reduces both model state and activation memory usage, though it may increase CPU overhead due to smaller per-GPU workloads.

#### Enable Tensor Parallelism

Configure the `tensor_model_parallel_size` parameter in your model configuration. Set this to greater than 1 to enable intra-layer model parallelism.

### Pipeline Parallelism

Pipeline Parallelism (PP) assigns consecutive layers or network segments to different GPUs, enabling each GPU to process different stages sequentially.

#### Enable Pipeline Parallelism

Set the `pipeline_model_parallel_size` parameter to a value greater than 1 to distribute layers across GPUs.

#### Interleaved Pipeline Schedule

This schedule divides computation on each GPU into multiple subsets of layers (model chunks) to minimize pipeline bubbles.

### Expert Parallelism

Expert Parallelism (EP) distributes experts of an MoE model across GPUs, affecting only expert layers while leaving other layers unchanged.

#### Enable Expert Parallelism

Set `expert_model_parallel_size` in your configuration. The number of experts should be divisible by this value.

## Activation Partitioning

These methods distribute activation memory across GPUs, crucial for training with large sequences or micro-batches.

### Sequence Parallelism

Sequence Parallelism (SP) distributes computing load and activation memory along the sequence dimension of transformer layers.

#### Enable Sequence Parallelism

Set `sequence_parallel=True` in your configuration, which requires tensor_model_parallel_size > 1.

### Context Parallelism

Context Parallelism (CP) partitions input tensors in the sequence dimension across all layers, unlike SP which operates on specific layers.

#### Enable Context Parallelism

Set `context_parallel_size` to a value greater than 1 to distribute sequence activations.

## Parallelism Configuration

When configuring parallel strategies in NeMo AutoModel:

1. Start with data parallelism (DDP or FSDP2) for basic scaling
2. Add tensor parallelism for memory-intensive models
3. Use pipeline parallelism for models with many layers
4. Consider activation partitioning techniques for large sequences
5. For MoE models, configure expert parallelism appropriately

The optimal configuration depends on your specific model architecture, hardware setup, and performance requirements. NeMo AutoModel provides flexible configuration options to tune these parameters for your use case.