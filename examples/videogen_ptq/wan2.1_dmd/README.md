

#docker to run the code on coreweave:

 srun --partition=interactive \
      --ntasks=1 \
      --nodes=1 \
      --time=02:00:00 \
      --gres=gpu:8  \
      --open-mode=append \
      --account=coreai_dlalgo_llm  \
      --job-name=coreai_dlalgo_llm-dfm \
      --container-mounts=/lustre/fsw/portfolios/coreai/users/linnanw:/linnanw \
      --container-image=nvcr.io/nvidian/pika:v0  \
      --export=ALL,MASTER_PORT=12345 \
      --pty bash

the preprocessed video data is located at: /lustre/fsw/portfolios/coreai/users/linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta

#Training script:
torchrun --nproc-per-node=8 main_t2v.py \
    --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta \
    --batch_size_per_node 1 \
    --learning_rate 1e-5 \
    --flow_shift 3.0 \
    --save_every 50 \
	--log_every 2 \
    --num_epochs 20 \
	--use_sigma_noise \
    --timestep_sampling uniform

Multinodes setup is still awaiting to test. You can reduce the epoch to 10 to save time


#Validation script

Make sure you passinto the checkpoint folder, and it is expected to be:

checkpoint-900:
consolidated_model.bin  optimizer  training_state.pt  transformer_model

Please make sure the consolidated_model.bin exists

python validate_t2v.py \
    --meta_folder /linnanw/hdvilla_sample/pika/wan21_codes/1.3B_meta  \
    --checkpoint /linnanw/wan2.1/wan_t2v_flow_outputs/checkpoint-300 \
    --output_dir ./t2v_finetuned_outputs \
	--num_frames 80