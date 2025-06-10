WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

python -m factscore.factscorer \
    --input_path ../../../data/labeled/InstructGPT.jsonl \
    --model_name retrieval+ChatGPT \
    --openai_key ../../../configs/openai_key.txt \
    --verbose 