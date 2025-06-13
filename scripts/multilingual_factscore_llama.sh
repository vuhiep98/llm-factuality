
WORKING_DIR=modules/MultilingualFactScore
cd ${WORKING_DIR}

CUDA_VISIBLE_DEVICES=0

# for LANG in es ar bn
for LANG in ar bn
do
    # python -m factscore.factscorer \
    #     --input_path ../../../data/annotation_wiki_all/${LANG}_gpt.jsonl \
    #     --model_name retrieval+llama \
    #     --output_path ../../outputs/multilingual_factscore \
    #     --verbose \
    #     --lang ${LANG} \
    #     --knowledge_source ${LANG}wiki \
    #     --use_atomic_facts
    
    python -m factscore.factscorer \
        --input_path ../../../data/annotation_wiki_all/${LANG}_gemini.jsonl \
        --model_name retrieval+llama \
        --output_path ../../outputs/multilingual_factscore \
        --verbose \
        --lang ${LANG} \
        --knowledge_source ${LANG}wiki \
        --use_atomic_facts
done