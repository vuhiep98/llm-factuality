from tqdm import tqdm
import json
import ast
import re
import csv
import random

gemini_path = "~/FActScore/data/to_annotate_data/bn/task/stage_3/gemini_2nd.json"
gpt4_path = "~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_2nd.json"

gemini_no = 23
gpt4_no = 22

# first_lst = []
# second_lst = []

with open(gemini_path) as f:
    for i, line in tqdm(enumerate(f)):
        gemini_dict = json.loads(line)
        # first_lst.append(dp)
        # first_name_lst.append(dp["topic"])
with open(gpt4_path) as f:
    for i, line in tqdm(enumerate(f)):
        gpt4_dict = json.loads(line)
print("Gemini dict")
for k in gemini_dict.keys():
    
    print(k, len(gemini_dict[k]))
print("GPT4 dict")
for k in gpt4_dict.keys():
    
    print(k, len(gpt4_dict[k]))

ratio_rarity_gemini = {"very freq": (3,3), "freq": (6, 5), "medium": (4,4), "rare": (3,4), "very rare": (5,4), "not overlap": (2,2)}
ratio_rarity_gpt4 = {"very freq": (3,3), "freq": (6, 5), "medium": (4,4), "rare": (3,4), "very rare": (5,4), "not overlap": (1,3)}

merge_gemini = ([], [])
merge_gpt4 = ([], [])

for k in gemini_dict.keys():
    merge_gemini[0].extend(gemini_dict[k][:ratio_rarity_gemini[k][0]])
    merge_gemini[1].extend(gemini_dict[k][-ratio_rarity_gemini[k][1]:])
    # print(k, len(gemini_dict[k]))
for k in gpt4_dict.keys():
    merge_gpt4[0].extend(gpt4_dict[k][:ratio_rarity_gpt4[k][0]])
    merge_gpt4[1].extend(gpt4_dict[k][-ratio_rarity_gpt4[k][1]:])
with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gemini_2nd_3_1.jsonl", 'w') as jsonl_file:
    for dictionary in merge_gemini[0]:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gemini_2nd_3_2.jsonl", 'w') as jsonl_file:
    for dictionary in merge_gemini[1]:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )

with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_2nd_3_1.jsonl", 'w') as jsonl_file:
    for dictionary in merge_gpt4[0]:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_2nd_3_2.jsonl", 'w') as jsonl_file:
    for dictionary in merge_gpt4[1]:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )
