WORKING_DIR=modules/FActScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

for LLM in InstructGPT ChatGPT PerplexityAI
do
    python -m factscore.factscorer \
        --input_path ../../../data/labeled/${LLM}.jsonl \
        --model_name retrieval+llama3 \
        --output_path /mnt/localssd/outputs/factscore \
        --verbose \
        --use_atomic_facts
done
