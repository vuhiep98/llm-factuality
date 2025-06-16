GPU=$1
LLM=$2

WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python -m factscore.factscorer \
    --input_path ../../../data/labeled/${LLM}.jsonl \
    --model_name ChatGPT+retrieval+llama3 \
    --output_path /mnt/localssd/outputs/factscore \
    --model_dir /mnt/localssd/.cache/factscore \
    --data_dir /mnt/localssd/.cache/factscore \
    --cache_dir /mnt/localssd/.cache/factscore \
    --verbose \
    --use_atomic_facts