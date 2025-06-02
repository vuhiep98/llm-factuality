import json
import time
import os

import sqlite3
import numpy as np
import pickle as pkl
from tqdm import tqdm
from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256
lang = "bn"
db_path = f"~/projects/factscore/real_evaluation/{lang}wiki_100.jsonl"
with open(db_path, 'r', encoding='utf-8') as file:
    sum_len = 0
    count = 0
    for i, line in tqdm(enumerate(file)):
        re_dict = json.loads(line)
        sum_len += len(re_dict["text"].split(SPECIAL_SEPARATOR))
        count += 1
print("AVG: ", sum_len/count)
