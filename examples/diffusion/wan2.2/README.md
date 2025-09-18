srun --partition=batch \
     --ntasks=1 \
     --nodes=1 \
     --time=04:00:00 \
     --gres=gpu:8  \
     --open-mode=append \
     --account=coreai_dlalgo_modelopt  \
     --job-name=coreai_dlalgo_modelopt-preprocess \
     --container-mounts=<YourPATH>:/<YourPATH> \
     --container-image=nvcr.io/nvidian/pika:v0  \
     --export=ALL,MASTER_PORT=12345 \
     --pty bash


export TP_SIZE=8
torchrun --nproc-per-node=8 wan22_t2v_tp_dp_pytorch.py


