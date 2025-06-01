# Calculate retrievied passages overlapped between different retrieval
import os
import json
import sqlite3
k_passage = 5
lang_1 = "en"
lang_2 = "bn"
instance_path = "~/translate_data/en_instances/en_instances/gpt4.jsonl"
labeled_path_1 = f"~/translate_data/en_instances/{lang_1}_instances/gpt4{lang_1}_retrieval+bloomz_factscore_output_provided_facts.json"
labeled_path_2 = f"~/translate_data/en_instances/{lang_2}_instances/gpt4{lang_2}_retrieval+bloomz_factscore_output_provided_facts.json"
lst_instance = []
with open(instance_path) as f:
    for line in f:
        lst_instance.append(json.loads(line).copy())

with open(labeled_path_1) as f:
    for line in f:
        dp_label_1 = json.loads(line)

decisions_1 = dp_label_1["decisions"]

with open(labeled_path_2) as f:
    for line in f:
        dp_label_2 = json.loads(line)

decisions_2 = dp_label_2["decisions"]
map_1_to_2 = {}
map_2_to_1 = {}

for i, (bio_1, bio_2) in enumerate(zip(decisions_1, decisions_2)):
    assert len(bio_1) == len(bio_2), (bio_1[0], bio_2[0])
    for fact_1, fact_2 in zip(bio_1,bio_2):
        # print(fact_1)
        name = "#".join([lst_instance[i]["topic"], lst_instance[i]["topic"]])
        map_1_to_2[" ".join([name, fact_1["atom"]])] = " ".join([name, fact_2["atom"]])
        map_2_to_1[" ".join([name, fact_2["atom"]])] = " ".join([name, fact_1["atom"]])
        # if "Jessie Mae Brown Beavers was born in 1908." == fact_1["atom"]:
        #     print("#".join([lst_instance[i]["topic"], fact_1["atom"]]))
        #     break
db_path = f"~/projects/factscore/{lang_1}wiki_150.db"
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
connection = sqlite3.connect(db_path, check_same_thread=False)
first_list = {}
retrieval_path = "~/projects/factscore/gtr-compare-new-retrieval-enwiki_150.json"
#retrieval_path = f"~/projects/factscore/compare-paraphrase-ar-new-arwiki_150.json"

#retrieval_path = f"~/projects/factscore/compare-{lang_1}-new-{lang_1}wiki_150.json"
cur_title = ""
retrieved_full = ""
lst_all_psg = []
lst_figure_1 = []
with open(retrieval_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_dict = json.loads(line)
    # print(json_dict)
    print("len(list(json_dict.keys()))", len(list(json_dict.keys())))
    for k in json_dict.keys():
        title = json_dict[k][0]["title"]
        # if title in ["Chris Cheney", "María Elena Medina-Mora Icaza", "Terence Blacker"]:
        #     continue
        if title != cur_title:
            cur_title = title
            cursor = connection.cursor()
            cursor.execute("SELECT text FROM documents WHERE title = ?", (cur_title,))
            results = cursor.fetchall()
            results = [r for r in results]
            retrieved_full = results[0][0]
            lst_all_psg = retrieved_full.split(SPECIAL_SEPARATOR)
            
            print("----------------------------------------------------------------")
            print("Current figure", cur_title)
        tmp = []
        for psg in json_dict[k]:
            tmp.append(lst_all_psg.index(psg["text"]))
        print(tmp, len(lst_all_psg))
        lst_figure_1.append((cur_title, len(lst_all_psg)))
        first_list[k] = (tmp, len(lst_all_psg), cur_title)
        #first_list.append((tmp, len(lst_all_psg), cur_title))

db_path = "~/projects/factscore/"+lang_2+"wiki_150.db"
SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
connection = sqlite3.connect(db_path, check_same_thread=False)
second_list = {} #compare-ar-new-arwiki_150
retrieval_path = f"~/projects/factscore/compare-paraphrase-{lang_2}-new-{lang_2}wiki_150.json"
retrieval_path = f"~/projects/factscore/compare-bm25-{lang_2}-new-{lang_2}wiki_150.json"

cur_title = ""
retrieved_full = ""
lst_all_psg = []
lst_figure_2 = []
with open(retrieval_path, 'r', encoding='utf-8') as file:
    for line in file:
        json_dict = json.loads(line)
    # print(json_dict)
    for k in json_dict.keys():
        title = json_dict[k][0]["title"]
        # if title in ["Chris Cheney", "María Elena Medina-Mora Icaza", "Terence Blacker"]:
        #     continue
        if title != cur_title:
            cur_title = title
            cursor = connection.cursor()
            cursor.execute("SELECT text FROM documents WHERE title = ?", (cur_title,))
            results = cursor.fetchall()
            results = [r for r in results]
            retrieved_full = results[0][0]
            lst_all_psg = retrieved_full.split(SPECIAL_SEPARATOR)
            
            # print("----------------------------------------------------------------")
            # print("Current figure", cur_title)
        tmp = []
        for psg in json_dict[k]:
            if psg["text"] == "":
                tmp.append(len(lst_all_psg))
                continue
            tmp.append(lst_all_psg.index(psg["text"]))
        # print(tmp, len(lst_all_psg))
        lst_figure_2.append((cur_title, len(lst_all_psg)))
        second_list[k] = (tmp, len(lst_all_psg), cur_title)
        #second_list.append((tmp, len(lst_all_psg), cur_title))
# for figure_1, figure_2 in zip(lst_figure_1, lst_figure_2):
#     print(figure_1)
#     print(figure_2)
#     print("-----------"*5)
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
count = 0
summ = 0
print(len(first_list.keys()), len(second_list.keys()))
for k in second_list.keys():
    k_1 = map_2_to_1[k]
    first = first_list[k_1]
    second = second_list[k]

    if (first[2] != second[2]):
        print("khac name", first[2], second[2])

    if first[1] < k_passage or first[2] != second[2] or len(first[0]) != len(second[0]) or len(first[0]) != len(set(first[0])): #or first[2] in ["George Cukor", "António de Oliveira Salazar", "Andreas Kisser"]
        # print(first[2], second[2])
        continue
    tmp = intersection(first[0], second[0])
    summ += len(tmp)/k_passage
    count += 1
    # summ += first[0][0] in second[0]
    # count += 1
    # if first[1] != second[1]:
        
    #     print(first, second,len(tmp)/5)
print(len(decisions_1), len(decisions_2))    
print(lang_2, count, summ, summ/count)