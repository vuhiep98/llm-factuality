GPU=$1
LLM=$2

WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=${GPU} python -m factscore.factscorer \
    --input_path ../../../data/labeled/${LLM}.jsonl \
    --model_name ChatGPT+retrieval+llama3 \
    --output_path ../../outputs/factscore \
    --model_dir .cache/factscore \
    --data_dir .cache/factscore \
    --cache_dir .cache/factscore \
    --verbose \
    --use_atomic_facts