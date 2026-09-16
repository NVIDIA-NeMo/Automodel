#!/bin/bash
# Tear the run down on every node and wait for the GPUs to drain.
#
# A failed rank does not end the run: the survivors sit in a collective holding all their
# memory until the 120-minute NCCL timeout, so on any OOM or traceback kill everything at
# once rather than waiting (RUNBOOK.md section 8.6). Ranks re-parent to PID 1
# and hold ~100 GB through SIGTERM, so this uses `docker rm -f` and then SIGKILL, and does
# not return until every GPU is under 2000 MiB and port 29500 is free -- otherwise the next
# launch dies on EADDRINUSE while the memory sampler records the previous run's peak.
set -uo pipefail

RUN_NAME=${RUN_NAME:-affine}
# Same order as NODE_IPS in launch_cluster.sh; re-check after every Spot re-provisioning.
# Every host listed here gets torchrun/nemo pkill -9, so trim it when other hosts are busy.
NODE_IPS=(10.30.0.2 10.30.0.4 10.30.0.3 10.30.0.7)

for rank in "${!NODE_IPS[@]}"; do
  ip=${NODE_IPS[$rank]}
  echo "--- tearing down rank $rank ($ip)"
  ssh -n -o BatchMode=yes -o ConnectTimeout=10 "chilaingoc@$ip" "
    sudo docker rm -f ${RUN_NAME}_node${rank} 2>/dev/null
    pkill -9 -f 'query-gpu=index,memory[.]used' 2>/dev/null  # bracket keeps pkill from matching its own shell
    sudo pkill -9 -f 'nemo_automodel.cli.app' 2>/dev/null
    sudo pkill -9 -f torchrun 2>/dev/null
    true"
done

echo "--- waiting for GPUs to drain and port 29500 to free"
for attempt in $(seq 1 60); do
  busy=0
  for ip in "${NODE_IPS[@]}"; do
    out=$(ssh -n -o BatchMode=yes -o ConnectTimeout=10 "chilaingoc@$ip" \
      'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1;
       ss -ltn 2>/dev/null | grep -c ":29500 "' 2>/dev/null)
    mem=$(echo "$out" | sed -n 1p); port=$(echo "$out" | sed -n 2p)
    if [ "${mem:-99999}" -ge 2000 ] || [ "${port:-1}" -ne 0 ]; then
      busy=1; echo "  $ip: max_gpu_mem=${mem} MiB port29500=${port}"
    fi
  done
  [ "$busy" -eq 0 ] && { echo "TEARDOWN_OK: all GPUs < 2000 MiB, port 29500 free"; exit 0; }
  sleep 10
done
echo "TEARDOWN_INCOMPLETE: something is still holding memory or the port" >&2
exit 1
