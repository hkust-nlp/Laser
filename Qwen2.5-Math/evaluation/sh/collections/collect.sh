
EVALDIR=$1
MODEL_NAME=$2
RUN_NAME=$3
TEMPERATURE=$4

RUN_NAME=${RUN_NAME}_t${TEMPERATURE}

python Qwen2.5-Math/evaluation/sh/collections/collect_results.py \
    --base_dir "$EVALDIR" \
    --model_name $MODEL_NAME \
    --wandb_project "openrlhf_sft_math-eval" \
    --wandb_api_key "${WANDB_API_KEY}" \
    --wandb_run_name $RUN_NAME \
    --benchmarks "gsm8k,math500,olympiadbench,aime24,amc23" \
    --use_wandb \
    --temperature $TEMPERATURE \
