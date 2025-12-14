MODEL_NAME=SubagentVL-7B-Fine-Restart-FixReward-80

API_KEY=killmeifyoucan
API_URL=http://localhost:18903/v1
VSTAR_BENCH_PATH=../data/vstar_bench
SAVE_PATH=./eval_results/vstar_deepeyes/subagent_fine_data_restart

NUM_WORKERS=8

EVAL_MODEL_URL=http://localhost:18901/v1
EVAL_API_KEY=killmeifyoucan
EVAL_MODEL_NAME=Qwen2.5-VL-7B-Instruct

python eval_subagent.py \
    --model_name ${MODEL_NAME} \
    --api_key ${API_KEY} \
    --api_url ${API_URL} \
    --vstar_bench_path ${VSTAR_BENCH_PATH} \
    --save_path ${SAVE_PATH} \
    --eval_model_name ${EVAL_MODEL_NAME} \
    --num_workers ${NUM_WORKERS}

python calc_score.py \
    --model_name ${MODEL_NAME} \
    --save_path ${SAVE_PATH} \
    --vstar_bench_path ${VSTAR_BENCH_PATH} \
    --eval_model_name ${EVAL_MODEL_NAME} \
    --api_key ${EVAL_API_KEY} \
    --api_url ${EVAL_MODEL_URL}