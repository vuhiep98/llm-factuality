import sqlite3
import time
import json
from tqdm import tqdm
lang = "es"
db_path = f"~/projects/factscore/real_evaluation/{lang}wiki.db"
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"

connection = sqlite3.connect(db_path, check_same_thread=False)
count_char = 0
list_translated_db = []
file_path_gemini = f"~/FActScore/data/to_evaluate/{lang}/gemini.jsonl"
file_path_gpt4 = f"~/FActScore/data/to_evaluate/{lang}/gpt4.jsonl"
name_list = []
with open(file_path_gpt4) as f:
        for line in f:
            dp = json.loads(line)
            name_list.append(dp["topic"])
with open(file_path_gemini) as f:
        for line in f:
            dp = json.loads(line)
            name_list.append(dp["topic"])
name_list = list(set(name_list))       
count = 0
start = time.time()
for title in tqdm(name_list):
    cursor = connection.cursor()
    cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
    results = cursor.fetchall()
    results = [r for r in results]
    # print(results)
    cursor.close()
    if len(results) == 0:
        continue
    print(title)
    results = {"title": title, "text": results[0][0]}
    # # [{"title": title, "text": para} for para in results[0][0]]
    # # print(results)
    count_char += len(results["text"])
    # text = translate_text("vi", results["text"]).replace("%%%%%%%%%%%%%%%%", SPECIAL_SEPARATOR)
    list_translated_db.append(results)
    count += 1
print("COUNT:", count)
print(time.time()-start)
print("COUNT_CHAR:", count_char)
jsonl_file_path = f"~/projects/factscore/real_evaluation/{lang}wiki_100.jsonl"
with open(jsonl_file_path, 'w') as jsonl_file:
    for dictionary in list_translated_db:
        json_line = json.dumps(dictionary, ensure_ascii=False)
        jsonl_file.write(json_line + '\n', )