WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

python -m factscore.factscorer \
    --input_path ../../data/labeled/InstructGPT.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/factscore \
    --verbose

python -m factscore.factscorer \
    --input_path ../../data/labeled/ChatGPT.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/factscore \
    --verbose

python -m factscore.factscorer \
    --input_path ../../data/labeled/PerplexityAI.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/factscore \
    --verbose