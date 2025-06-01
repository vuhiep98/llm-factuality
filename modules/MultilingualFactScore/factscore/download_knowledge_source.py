from datasets import load_dataset
from tqdm import tqdm
from collections import Counter, defaultdict
import os
import gzip
import json
import csv
langs = ["es", "ar", "bn"]
# with open('~/factscore/NPM/studied_langs.csv', 'r', newline='') as csvfile:
#     # Create a CSV reader object
#     csvreader = csv.reader(csvfile)
    
#     # Iterate over each row in the CSV file
#     for row in csvreader:
#         # Process each row as needed
#         print(row)
#         langs.append(row[0])
for lang in langs:
    print(lang)
    dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}")
    # output_dir = "~/projects/factscore/real_evaluation/"
    output_dir = "../../.cache/factscore"
    lst_all = []
    count_sub = 0
    count = 0
    ds = dataset["train"]
    for e in tqdm(ds):
#         if count == num_docs:
#             break
#         if e["timestamp"] == "":
#             print(e)
#             continue
        lst_all.append(e)
        count += 1
    os.makedirs(output_dir, exist_ok=True)
    output_json_file = os.path.join(output_dir, f'{lang}wiki.json')
    with open(output_json_file, 'w') as f:
        for e in lst_all:
            json_line = json.dumps(e, ensure_ascii=False)
            f.write(json_line + '\n', )
    