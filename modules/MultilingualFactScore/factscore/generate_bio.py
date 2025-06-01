from factscore.openai_lm import OpenAIModel
import json
import os
from tqdm import tqdm
import google.generativeai as genai
from factscore.gemini_lm import GeminiModel

path = "~/FActScore/data/entities/chosen_entities/es/es_all.json"
lm = OpenAIModel("ChatGPT-gpt4", cache_file="~/factscore/GPT-4-gen-bio.pkl", key_path="~/projects/open_ai_key.txt")
# gemini_lm = GeminiModel("Gemini", cache_file=os.path.join("~/projects/factscore", "Gemini-pro.pkl"), key_path="~/projects/open_ai_key.txt")
# genai.configure(api_key="AIzaSyCNpGMai0auQpfuEXN847FAoOe2aYMBOks")
# gemini_lm = GeminiModel("Gemini", cache_file="~/projects/factscore/Gemini-pro.pkl", key_path="~/projects/open_ai_key.txt")
with open(path) as f:
    print("Entity path", path)
    lang = path.split("/")[-1].split("_")[0]
    print("lang", lang)
    for line in f:
        dp = json.loads(line)
        break
    lst_save = []
    for k in dp.keys():
        print("######################")
        print(k)
        
        for i, e in tqdm(enumerate(dp[k])):
            save = {"input": e["query"], "output": "", "topic": e["entity"], "cat": [k, e["region"]], "lang": lang, "link": e["link"]}
            # save["output"], log = gemini_lm.generate(save["input"])
            save["output"], _= lm.generate(save["input"])
            print("entity:", save["topic"])
            print("output:", save["output"])
            lst_save.append(save.copy())
            # break
            # if i%5:
            #     lm.save_cache()
            # break
        # break
#         lm.save_cache()
# lm.save_cache()
# gemini_lm.save_cache()
output_json_file_path = "~/FActScore/data/to_annotate_data/es/gen_bio.jsonl"
with open(output_json_file_path, 'w') as jsonl_file:
    for dictionary in lst_save:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )