#!/bin/bash

WORK_DIR="/path/to/your/work/dir"

export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH=$PYTHONPATH:$WORK_DIR
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

MASTER_NODE=${SLURM_NODELIST%%,*} 
MASTER_PORT=31012
DASHBOARD_PORT=31010

echo "SLURM_NODEID: ${SLURM_NODEID}"
echo "SLURM_PROCID: ${SLURM_PROCID}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
echo "MASTER_NODE: ${MASTER_NODE}"

IP_FILE="multi_log/master_ip_${SLURM_NODELIST}.txt"

echo "IP_FILE: ${IP_FILE}"

mkdir -p multi_log

# Create custom Ray temporary directory
RAY_TMP_DIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
mkdir -p $RAY_TMP_DIR
chmod 755 $RAY_TMP_DIR
export RAY_TMPDIR=$RAY_TMP_DIR
echo "Using Ray temporary directory: $RAY_TMP_DIR"

if [ "${SLURM_PROCID}" -eq 0 ]; then
    echo "Starting Ray head node on ${MASTER_NODE}"
    
    ip_addr=$(hostname -I | awk '{print $1}')
    echo ${ip_addr} > ${IP_FILE}
    echo "Master IP: ${ip_addr}"
    
    ray start --head \
        --node-ip-address=${ip_addr} \
        --port=${MASTER_PORT} \
        --num-gpus=${SLURM_GPUS_ON_NODE} \
        --dashboard-port=${DASHBOARD_PORT} \
        --min-worker-port=32002 \
        --max-worker-port=33001 \
        --dashboard-agent-listen-port=52365 \
        --ray-client-server-port=52097 \
        --temp-dir=${RAY_TMP_DIR} \
        --block &
    
    sleep 10
    
    echo "Checking Ray status on head node..."
    ray status
    
    echo "Submitting Ray job..."
    ray job submit -v \
        --address="http://127.0.0.1:${DASHBOARD_PORT}" \
        --runtime-env-json="{
            \"working_dir\": \"/project/deemreason/weiliu/verl\",
            \"excludes\": [\".git/**\", \"wandb/**\", \"logs/**\", \"data/**\", \"Qwen2.5-Math/evaluation/**\"],
            \"temp_dir\": \"${RAY_TMP_DIR}\"
        }" \
        -- bash $1

else
    echo "Starting Ray worker node on $(hostname)"
    

    while [ ! -f ${IP_FILE} ]; do
        echo "Waiting for master IP file..."
        sleep 5
    done
    
    master_ip=$(cat ${IP_FILE})
    echo "Found master IP: ${master_ip}"
    
    sleep 20
    
    ray start \
        --address="${master_ip}:${MASTER_PORT}" \
        --num-gpus=${SLURM_GPUS_ON_NODE} \
        --block
fi
