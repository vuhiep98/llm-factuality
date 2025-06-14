GPU=$1
LLM=$2

WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python -m factscore.factscorer \
    --input_path ../../../data/labeled/${LLM}.jsonl \
    --model_name ChatGPT+retrieval+llama2 \
    --output_path /mnt/localssd/outputs/factscore \
    --verbose \
    --use_atomic_facts