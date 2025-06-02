python -m factscore.factscorer \
    --input_path ../../data/annotation_wiki_all/es_gpt4.jsonl \
    --model_name retrieval+llama \
    --knowledge_source eswiki \
    --n_samples 2 \
    --lang es