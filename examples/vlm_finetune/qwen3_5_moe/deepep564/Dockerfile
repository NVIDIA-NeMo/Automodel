# nemo-automodel 26.08.00 with two changes needed for the 2-node v4_88k runs:
#
# 1. DeepEP 4214430 (the hybrid-ep pin the image ships) + DeepEP PR #564 backported.
#    #564 fixes internode normal dispatch against NVSHMEM >= 3.5 (device-state RDC and the
#    RC QP layout). Without it, EP16 across two nodes faults on the first internode
#    dispatch (Xid 31 MMU fault @ 0x0 on every GPU, then "DeepEP error: timeout
#    (dispatch CPU)"). Same defect as NVIDIA-NeMo/RL issue #4027. The patch also carries
#    docker/common/deepep.patch (CCCL include path, nvidia-ml-py), which the build needs.
#    HYBRID_EP_MULTINODE=0 skips HybridEP's DOCA/NCCL-submodule path (unused with
#    dispatcher=deepep).
#
# 2. torchao pinned to 0.14.0. The image's 0.17.0 fails on FSDP2 DTensor params inside
#    torchao.optim.AdamW8bit's compiled step ("OptimState8bit dispatch: unimplemented
#    aten.view(dtype)"); 0.14.0 is the version the 8xH200 sweep validated. --no-deps keeps
#    the image's torch.
#
# Build (on every node):
#   docker build -t nemo-automodel:26.08.00-deepep564 examples/vlm_finetune/qwen3_5_moe/deepep564
FROM nvcr.io/nvidia/nemo-automodel:26.08.00

ARG DEEPEP_COMMIT=42144303752422ade37f24bca9e2dde12df70e09
COPY deepep_4214430_backport564.patch /opt/deepep_backport564.patch
RUN apt-get update && apt-get install -y --no-install-recommends libnvidia-ml-dev && \
    git clone https://github.com/deepseek-ai/DeepEP.git /opt/DeepEP-564 && \
    cd /opt/DeepEP-564 && git checkout ${DEEPEP_COMMIT} && \
    git apply /opt/deepep_backport564.patch && \
    HYBRID_EP_MULTINODE=0 TORCH_CUDA_ARCH_LIST="9.0" MAX_JOBS=64 \
      pip install --no-cache-dir --no-build-isolation --no-deps -v . && \
    apt-get purge -y libnvidia-ml-dev && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* && \
    python -c "import importlib.metadata as m; print('deep_ep', m.version('deep_ep'))"

RUN pip install --no-cache-dir --no-deps "torchao==0.14.0" && \
    python -c "import importlib.metadata as m; from torchao.optim import AdamW8bit; print('torchao', m.version('torchao'))"
