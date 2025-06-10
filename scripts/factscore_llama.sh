WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

# for LLM in InstructGPT ChatGPT PerplexityAI
for LLM in InstructGPT
do
    python -m factscore.factscorer \
        --input_path ../../../data/labeled/${LLM}.jsonl \
        --model_name retrieval+llama \
        --output_path ../../outputs/factscore \
        --verbose > ../../outputs/logs/factscore_${LLM}_log_llama2.txt
done