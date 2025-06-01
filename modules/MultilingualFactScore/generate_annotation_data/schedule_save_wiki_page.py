from itertools import product
import os
from tqdm import tqdm
import json
import subprocess

# all_subjects = [
#     "alpaca-7B","alpaca-13B","alpaca",
#     "ChatGPT","dolly-12b","gpt4",
#     "InstructGPT", "mpt-7b", "pythia-12b", "stablelm-alpha-7b", "vicuna-7b", "vicuna-13b"

# ]

# all_evaluators = [
#     'bloomz',
#     'mistral'
# ]


def schedule(path, parent_dir):
    output_dir = os.path.join(parent_dir, "perplexity_reference_en")
    os.makedirs(output_dir, exist_ok=True)
    assert os.path.exists(output_dir), "Dir not exists "+ output_dir
    cmd = ""
    with open(path) as f:
        for i, line in tqdm(enumerate(f)):
            dp = json.loads(line)
            wiki_link = dp["link"]
            if wiki_link == "":
                continue
            name = "_".join(dp["topic"].split(" "))
            path_output = f"{output_dir}/{name}.txt"
            cmd += "python3 -m factscore.get_wiki_page " + \
                f" --wiki_link '{wiki_link}' " + \
                f" --output_file '{path_output}'\n"
    with open('~/FActScore/scripts/get_wiki_page.sh','w') as f:
        f.write(cmd)

    # for evaluator, subject in product(all_evaluators, all_subjects):
    #     output_dir=f"data/output/{lang}_mistral_generated_facts/all_reasoning_retrieval_{evaluator}"
    #     os.makedirs(output_dir, exist_ok=True)

    #     wiki_link=f"{input_dir}/{subject}.jsonl"
    #     path_output=f"{output_dir}/{subject}.txt"

    #     assert os.path.exists(path_input), "File not exists "+ path_input

    #     cmd = "python3 -m factscore.get_wiki_page " + \
    #         f" --wiki_link {path_input} " + \
    #         f" --output_file {path_output}\n"
        
    #     with open('jobs.txt','a') as f:
    #         f.write(cmd)


if __name__ == "__main__":
    # lang = "bn"
    # stage = "stage_1"
    # model = "gemini"
    for lang in ["bn"]:
        for stage in ["stage_3"]:
            for model in ["gpt4"]:
                for annotator in ["regenerate"]:
                    #~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_regenerate_have_facts.jsonl
                    # t = f"~/FActScore/data/to_annotate_data/es/task/sub_task_1/{model}_{annotator}_have_facts.jsonl"
                    schedule(f"~/FActScore/data/to_annotate_data/{lang}/task/{stage}/{model}_{annotator}_have_facts.jsonl", f"~/FActScore/data/to_annotate_data/{lang}/task/{stage}/{annotator}/working_files")
                    subprocess.run(["bash", "~/FActScore/scripts/get_wiki_page.sh"]) 
                    # for sub_stage in ["3_1", "3_2"]:
                    #     t = f"~/FActScore/data/to_annotate_data/es/task/stage_3/{model}_{annotator}_{sub_stage}_have_facts.jsonl"
                    #     schedule(f"~/FActScore/data/to_annotate_data/{lang}/task/stage_3/{model}_{annotator}_{sub_stage}_have_facts.jsonl", f"~/FActScore/data/to_annotate_data/{lang}/task/{stage}/{sub_stage}/{annotator}/working_files")
                    #     subprocess.run(["bash", "~/FActScore/scripts/get_wiki_page.sh"]) 
    
    # # schedule('es')
    # schedule('fr')
    # schedule('ru')
    # # schedule('hi')
    # # schedule('vi')
    
    # schedule('vi')
