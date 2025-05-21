WORKING_DIR="/path/to/your/work/dir"
HEAD_IP=""
HEAD_PORT=""

RUN_SCRIPT=$1

ray job submit --address=${HEAD_IP}:${HEAD_PORT} \
  --entrypoint-num-cpus=1 \
  --runtime-env-json='{
        "working_dir": "'${WORKING_DIR}'",
        "env_vars": {
          "http_proxy": "",
          "https_proxy": ""
        }
    }' -- bash $RUN_SCRIPT