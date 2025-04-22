#!/bin/bash
AGENT_ROOT="${AGENT_ROOT:-/datacanvas/titan_cv_agent_sandbox}"
ENV_ROOT="${ENV_ROOT:-/datacanvas/titan_cv_agent_sandbox/envs}"
LOG_ROOT=${AGENT_ROOT}/base/logs
#OUTPUT_ROOT="${BASE_LOCAL_OUTPUT_PREFIX:-AGENT_ROOT/}"

mkdir -p "$LOG_ROOT"

echo "Startup Redis Server..."
nohup redis-server > ${AGENT_ROOT}/base/logs/redis.log 2>&1 &

echo "Startup Workers..."
conda run --no-capture-output -p ${ENV_ROOT}
cd ${AGENT_ROOT}/base
nohup celery -A celery_worker.app worker --loglevel=info --concurrency=2 -Q cpu_queue > ${LOG_ROOT}/cpu_queue.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup celery -A celery_worker.app worker --loglevel=info --concurrency=2 -Q gpu_light_queue > ${LOG_ROOT}/gpu_light_queue.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup celery -A celery_worker.app worker --loglevel=info --concurrency=2 -Q gpu_heavy_queue > ${LOG_ROOT}/gpu_heavy_queue.log 2>&1 &

echo "Startup Flask..."
cd ${AGENT_ROOT}/base
python api.py
echo "All services have been started. Check log files for details."
