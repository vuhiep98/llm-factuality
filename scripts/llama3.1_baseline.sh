CUDA_VISIBLE_DEVICES=0 \
    python -m src.baseline \
    --input_file ../data/labeled/InstructGPT.jsonl \
    --model_path meta-llama/Llama-3.1-8B-Instruct