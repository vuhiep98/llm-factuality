from tqdm import tqdm
import json
import ast
import re
import csv
import random

first_path = "~/FActScore/data/to_annotate_data/bn/gemini.jsonl"
second_path = "~/FActScore/data/to_annotate_data/bn/gpt4.jsonl"
first_lst = []
second_lst = []

with open(first_path) as f:
    for i, line in tqdm(enumerate(f)):
        dp = json.loads(line)
        first_lst.append(dp)
with open(second_path) as f:
    for i, line in tqdm(enumerate(f)):
        dp = json.loads(line)
        second_lst.append(dp)
first_name_lst = [e["topic"] for e in first_lst]
second_name_lst = [e["topic"] for e in second_lst]
count = 0
overlap = 0
overlap_name_lst = []
assert len(first_name_lst) == len(set(first_name_lst))
assert len(second_name_lst) == len(set(second_name_lst))
assert len(first_name_lst) == len(second_name_lst)
not_overlap_in_first_name = []
for e in first_name_lst:
    count += 1
    if e in second_name_lst:
        overlap += 1
        overlap_name_lst.append(e)
    else:
        not_overlap_in_first_name.append(e)
not_overlap_in_first = [e for e in first_lst if e["topic"] in not_overlap_in_first_name]
not_overlap_in_second_name = []
for e in second_name_lst:
    if e not in first_name_lst:
        not_overlap_in_second_name.append(e)
not_overlap_in_second = [e for e in second_lst if e["topic"] in not_overlap_in_second_name]
overlap_lst = ([e for e in first_lst if e["topic"] in overlap_name_lst], [e for e in second_lst if e["topic"] in overlap_name_lst])
rarity_instances = ({"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": []}, {"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": []})
for e in overlap_lst[0]:
    rarity_instances[0][e["cat"][0]].append(e)
for e in overlap_lst[1]:
    rarity_instances[1][e["cat"][0]].append(e)
rarity_instances[0]["not overlap"] = not_overlap_in_first
rarity_instances[1]["not overlap"] = not_overlap_in_second
for k in rarity_instances[0].keys():
    print(k, len(rarity_instances[0][k]), len(rarity_instances[1][k]))

ratio_with_rarity = ({"very freq": (6, 6), "freq": (11, 11), "medium": (8,8), "rare": (8, 7), "very rare": (8, 9), "not overlap": (4, 4)}, {"very freq": (6, 6), "freq": (11, 11), "medium": (8, 8), "rare": (8, 7), "very rare": (8, 9), "not overlap": (4, 4)})

partition = ([{"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": [], "not overlap": []}, {"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": [], "not overlap": []}], [{"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": [], "not overlap": []}, {"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": [], "not overlap": []}])

for k in partition[0][0].keys():
    rarity_instances[0][k].sort(key=lambda x:x["topic"])
    partition[0][0][k] = rarity_instances[0][k][:ratio_with_rarity[0][k][0]]
    partition[0][1][k] = rarity_instances[0][k][-ratio_with_rarity[0][k][1]:]
    rarity_instances[1][k].sort(key=lambda x:x["topic"])
    partition[1][0][k] = rarity_instances[1][k][:ratio_with_rarity[1][k][0]]
    partition[1][1][k] = rarity_instances[1][k][-ratio_with_rarity[1][k][1]:]

count_1st = 0
count_2nd = 0
for k in partition[0][0].keys():
    print("1st", k, len(partition[0][0][k]))
    print("2nd", k, len(partition[0][1][k]))
    count_1st += len(partition[0][0][k])
    count_2nd += len(partition[0][1][k])
print(count_1st)
print(count_2nd)
merge_1st = []
merge_2nd = []

for k in partition[0][0].keys():
    partition[0][0][k].sort(key=lambda x:x["topic"])
    partition[0][1][k].sort(key=lambda x:x["topic"])

with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gemini_1st.json", 'w') as jsonl_file:
    json_line = json.dumps(partition[0][0], ensure_ascii=False)
    jsonl_file.write(json_line + '\n', )
with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gemini_2nd.json", 'w') as jsonl_file:
    json_line = json.dumps(partition[0][1], ensure_ascii=False)
    jsonl_file.write(json_line + '\n', )

count_1st = 0
count_2nd = 0
for k in partition[1][0].keys():
    print("1st", k, len(partition[1][0][k]))
    print("2nd", k, len(partition[1][1][k]))
    count_1st += len(partition[1][0][k])
    count_2nd += len(partition[1][1][k])
print(count_1st)
print(count_2nd)
merge_1st = []
merge_2nd = []

for k in partition[1][0].keys():
    partition[1][0][k].sort(key=lambda x:x["topic"])
    partition[1][1][k].sort(key=lambda x:x["topic"])

with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_1st.json", 'w') as jsonl_file:
    json_line = json.dumps(partition[1][0], ensure_ascii=False)
    jsonl_file.write(json_line + '\n', )
with open("~/FActScore/data/to_annotate_data/bn/task/stage_3/gpt4_2nd.json", 'w') as jsonl_file:
    json_line = json.dumps(partition[1][1], ensure_ascii=False)
    jsonl_file.write(json_line + '\n', )











# ####################################
# second_partition = [{"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": []}, {"very freq": [], "freq": [], "medium": [], "rare": [], "very rare": []}]
# for k in second_partition[0].keys():
#     second_partition[0][k] = rarity_instances[k][:ratio_with_rarity[k][0]]
#     second_partition[1][k] = rarity_instances[k][-ratio_with_rarity[k][1]:]
# second_partition[0]["not overlap"] = rarity_instances["not overlap second"][:ratio_with_rarity["not overlap second"][0]]
# second_partition[1]["not overlap"] = rarity_instances["not overlap second"][-ratio_with_rarity["not overlap second"][1]:]


# count_1st = 0
# count_2nd = 0
# for k in second_partition[0].keys():
#     print("1st", k, len(second_partition[0][k]))
#     print("2nd", k, len(second_partition[1][k]))
#     count_1st += len(second_partition[0][k])
#     count_2nd += len(second_partition[1][k])
# print("count_1st", count_1st)
# print("count_2nd", count_2nd)

# merge_1st = []
# merge_2nd = []

# for k in second_partition[0].keys():
#     # print("1st", k, len(first_partition[0][k]))
#     # print("2nd", k, len(first_partition[1][k]))
#     # count_1st += len(first_partition[0][k])
#     # count_2nd += len(first_partition[1][k])
#     merge_1st.extend(second_partition[0][k])
#     merge_2nd.extend(second_partition[1][k])
# merge_1st = random.shuffle(merge_1st)
# merge_2nd = random.shuffle(merge_2nd)
