#!/bin/bash
#SBATCH -A coreai_dlalgo_llm
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node=8
#SBATCH --time 00:30:00
#SBATCH --exclusive
#SBATCH --output=./CICD_nightly_RUN_slurm_%x_%j.out
#SBATCH --error=./CICD_nightly_RUN_slurm_%x_%j.err
#SBATCH -J Automodel_Diffusion

# Multi-node env
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export NUM_GPUS=8
export WORLD_SIZE=$(($NUM_GPUS * $SLURM_NNODES))

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0


# Experiment env
# TODO: update the key
export WANDB_API_KEY=""
export HF_HOME="/linnanw/hdvilla_sample/cache"
export HF_TOKEN=""


# SHARED paths on Lustre (visible to ALL nodes)
# TODO: update the path
UV_SHARED_DIR="/lustre/fsw/portfolios/coreai/users/linnanw/uv_cache/${SLURM_JOB_ID}"

# Step 1: Pre-build on a SINGLE node first (avoids race conditions)
# Create a shared venv on LUSTRE that ALL nodes can access
read -r -d '' PREBUILD_CMD <<EOF
cd /opt/Automodel/
echo "=== Pre-building on single node ==="
mkdir -p ${UV_SHARED_DIR}
export UV_CACHE_DIR=${UV_SHARED_DIR}/cache
export UV_PROJECT_ENVIRONMENT=${UV_SHARED_DIR}/.venv
# Sync creates the venv and installs all packages (including building local packages)
uv sync
echo "=== Pre-build complete ==="
echo "Venv created at: ${UV_SHARED_DIR}/.venv"
ls -la ${UV_SHARED_DIR}/.venv/bin/python
EOF
echo "$PREBUILD_CMD"

# TODO: the container image should be updated, also the container-mounts
echo "Running pre-build step on single node..."
srun \
    -N 1 \
    --ntasks=1 \
    --mpi=pmix \
    --container-entrypoint \
    --no-container-mount-home \
    --container-image=YOUR_CONTAINER_IMAGE \
    --container-mounts=/lustre:/lustre,/lustre/fsw/portfolios/coreai/users/linnanw:/linnanw,PATH_TO_AUTOMODEL:/opt/Automodel/ \
    --export=ALL \
    bash -c "$PREBUILD_CMD"

# Step 2: Now run on all nodes using the SAME pre-built venv on Lustre
read -r -d '' CMD <<EOF
cd /opt/Automodel/; whoami; date; pwd;
# Activate the pre-built venv
echo "Activating venv at: ${UV_SHARED_DIR}/.venv"
source ${UV_SHARED_DIR}/.venv/bin/activate
# CRITICAL: Set PYTHONPATH so that even when torchrun spawns workers using
# /usr/bin/python directly (bypassing venv symlink), they still find packages
export PYTHONPATH="${UV_SHARED_DIR}/.venv/lib/python3.12/site-packages:\${PYTHONPATH}"
echo "PYTHONPATH: \$PYTHONPATH"
which python
python -c "import nemo_automodel; print('nemo_automodel OK')"
# Now torchrun workers will find packages via PYTHONPATH
torchrun --nnodes=\$SLURM_NNODES --nproc-per-node=\$NUM_GPUS --rdzv_backend=c10d --rdzv_endpoint=\$MASTER_ADDR:\$MASTER_PORT examples/diffusion/pretrain/pretrain.py  -c examples/diffusion/pretrain/cicd/wan21_cicd_nightly_video.yaml
EOF
echo "$CMD"

echo "Running training on all nodes..."
srun \
    --mpi=pmix \
    --container-entrypoint \
    --no-container-mount-home \
    --container-image=YOUR_CONTAINER_IMAGE \
    --container-mounts=/lustre:/lustre,/lustre/fsw/portfolios/coreai/users/linnanw:/linnanw,PATH_TO_AUTOMODEL:/opt/Automodel/ \
    --export=ALL \
    bash -c "$CMD"
