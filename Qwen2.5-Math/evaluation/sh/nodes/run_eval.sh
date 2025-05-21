BASE_STORAGE_DIR="/PATH/TO/BASE/STORAGE/DIR"

RUNNAME=$1
INIT_MODEL_PATH=$2
TPSIZE=${3:-1}
STEPS=${4:-""}
NSAMPLING=16
TOPP=0.95
TEMPLATE="deepseek-r1-system-boxed"

BENCHMARKS="aime24,amc23,math500,olympiadbench"

bash Qwen2.5-Math/evaluation/sh/nodes/eval_math_nodes.sh \
    --run_name $RUNNAME \
    --init_model $INIT_MODEL_PATH \
    --template $TEMPLATE  \
    --tp_size $TPSIZE \
    --add_step_0 false  \
    --temperature 0.6 \
    --top_p $TOPP \
    --max_tokens 32768 \
    --output_dir $OUTPUT_DIR \
    --benchmarks $BENCHMARKS \
    --n_sampling $NSAMPLING \
    --specific_steps $STEPS \
    --base_storage_dir $BASE_STORAGE_DIR \
    --add_step_0 true \