WORKING_DIR=modules/MultilingualFactScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

python -m factscore.factscorer \
    --input_path ../../../data/annotation_wiki_all/es_gpt4.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/multilingual_factscore \
    --verbose \
    --lang es \
    --knowledge_source eswiki \
    --use_atomic_facts

python -m factscore.factscorer \
    --input_path ../../../data/annotation_wiki_all/ar_gpt4.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/multilingual_factscore \
    --verbose \
    --lang ar \
    --knowledge_source arwiki \
    --use_atomic_facts

python -m factscore.factscorer \
    --input_path ../../../data/annotation_wiki_all/bn_gpt4.jsonl \
    --model_name retrieval+llama \
    --output_path ../../outputs/multilingual_factscore \
    --verbose \
    --lang bn \
    --knowledge_source bnwiki \
    --use_atomic_facts