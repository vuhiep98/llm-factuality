python -m factscore.factscorer \
    --input_path factscore_data/unlabeled/Vicuna-13B.jsonl \
    --model_name retrieval+ChatGPT \
    --openai_key ../../configs/openai_key.txt \
    --verbose 