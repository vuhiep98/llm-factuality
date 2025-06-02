import sqlite3
import time
import json
from tqdm import tqdm
lang = "bn"
db_path = f"~/projects/factscore/real_evaluation/{lang}wiki_100.db"
data_path = f"~/projects/factscore/real_evaluation/{lang}wiki_100.jsonl"
connection = sqlite3.connect(db_path, check_same_thread=False)
cursor = connection.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
c = connection.cursor()
c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
titles = set()
output_lines = []
tot = 0
start_time = time.time()
with open(data_path, "r") as f:
    for line in f:
        print(line)
        dp = json.loads(line)
        title = dp["title"]
        text = dp["text"]
        if title in titles:
            continue
        titles.add(title)

        output_lines.append((title, text.strip()))
        tot += 1
c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
connection.commit()
connection.close()