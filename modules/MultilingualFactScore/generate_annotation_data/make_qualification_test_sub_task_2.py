# Data to be written to the file
from tqdm import tqdm
import json
import ast
import re
import csv
import os
lang = "es"
annotator = "2"
question_to_write = """
            Question #1: Is the fact factually correct according to (supported by) your language Wikipedia? Refer to instructions within Step 2.1 and Tips of the provided rule.
                A. Supported.
                B. Not-supported.
                C. Irrelevant.
            Question #2: Provide supporting evidences of your Question #1's answer or edit if the is not supported. Refer to instructions within Step 2.2, 2.3 and Tips of the provided rule.
                Answer:
            Question #3: Is the fact factually correct according to (supported by) English Wikipedia? Similar to question #0 but with different sources.
                A. Supported.
                B. Not-supported.
                C. Irrelevant.
            Question #4: Provide supporting evidences of your Question #3's answer or edit if the is not supported. Similar to question #1 but with different sources.
                Answer:
            Question #5: Is the fact factually correct according to (supported by) the Internet? Similar to question #0 but with different sources.
                A. Supported.
                B. Not-supported.
                C. Irrelevant.
            Question #6: Provide supporting evidences of your Question #5's answer or edit if the is not supported. Similar to question #1 but with different sources.
                Answer:
"""
biography_quick_review = """
- Question #0: Is the given biography relevant, irrelevant to the corresponding figure or an abstain generation?
    A. Relevant.
    B. Irrelevant.
    C. Abstain.      
"""
data_to_write = """--------------------Sub-task 2 - Fact extraction--------------------
--------------------
--------------------
--------------------INTRODUCTION--------------------
- With each instance of this task, you will be given a sentence and FOUR lists of facts extracted from that sentence by FOUR different models. You are supposed grade those FOUR extraction by a number of pre-defined criteria.

- EVALUATION CRITERIA:
    + For each sentence and the corresponding extraction from one out of four models, 10 points will be given. Points will be subtracted or added based on the following criteria
        - If the extraction NEED MERGE (a sentence is excessively extracted and not verifiable), 1.25 point will be subtracted from its total grade (-1.25 point) 
            Example: "He played", "He did", "In 2009." are excessively extracted facts from sentences, too vague and not verifiable.
        - If the extraction NEED SPLIT (a sentence is atomic enough), 1.25 point will be subtracted from its total grade (-1.25 point)
            Example: The list of extracted facts includes "He played for Real Madrid in 2010" but doesn't include "He played for Real Madrid", but if the list doesn't include "He played in 2010", it could be tolerable.
        - If the extraction contains DUPLICATED extracted facts, 2 point will be subtracted from its total grade (-2 point)
            Example: The list of extracted facts includes more than one "He played for Real Madrid".
        - If the extraction MISSED information within the original sentence (Extracted facts do not adequately or cover information within the original sentence), 2 point will be subtracted from its total grade (-2 point)
            Example: The original sentence is "He played for Real Madrid in 2010", but extracted facts do not cover details about the year when the event happened.
        - If information within sentence is corrupted, falsely modified (with respect to the original sentence, not factuality), 2 point will be subtracted from its total grade (-2 point)
            Example: The sentence "He played for Real Madrid in 2010", but "He played for Real Betis" is one of the extracted facts.
        - The quality of extraction in linguistic perspective, tell whether there are any linguistic errors (grammar, spelling or coherence).
            + Five level:
                - 5: Excellent, there are no errors. (0 point subtracted, -0)
				- 4: Good, there are negligible or small quantities of acceptable errors. (0 point subtracted, -0)
				- 3: Average, there are noticeable errors but have little effect on sentence understanding and factual evaluating. (0.5 point subtracted, -0.5)
				- 2: Fair, there are a considerable number of errors but it is generally possible to get pieces of information from extracted facts. (1 point subtracted, -1)
				- 1: Poor, Errors within extracted facts hugely hinder fact comprehension and evaluation (cannot get the piece of information delivered by the fact). (1.5 point subtracted, -1.5)
        - There is an additional column where you can give bonus or subtracted points based on criterias that are not specified above and provide reason in comment column (You can subtract or add up to 1.5 point for this column)
    + For each non-default grading for each above criteria (non-zero), you should provide the reason on the corresponding "Comment" columns.

- Please submit your answer through the provided answer sheet

------------EXAMPLE------------

1. SENTENCE:
    1.a. Facts extracted by Model 1:
        -
        -
        -
        -

    1.b. Facts extracted by Model 2:
        -
        -
        -
        -
    1.c. Facts extracted by Model 3:
        -
        -
        -
        -
    1.d. Facts extracted by Model 4:
        -
        -
        -
        -
------------END OF EXAMPLE------------
--------------------END OF INTRODUCTION--------------------

"""
data_to_write_csv = []
alphab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "h", "y", "z"]
data_to_write_csv = []

json_path = f"~/FActScore/data/to_annotate_data/{lang}/task/sub_task_2/sent2fact_{annotator}.json"

data_to_write_csv = []
with open(json_path) as f:
    for line in f:
        sent2facts = json.loads(line)
for i, sent in enumerate(list(sent2facts.keys())):
    data_to_write += "{}. SENTENCE: {}\n".format(i+1, sent)
    data_to_write_csv.append(["{}. SENTENCE: {}\n".format(i+1, sent), "Need merge", "Comment", "Need split", "Comment", "Duplicated", "Comment", "Inadequate covering", "Comment", "Need Edit", "Comment", "Linguistics", "Comment", "Bonus", "Comment", "Total grade"])
    # print(f"{i}. SENTENCE: {sent}")
    for j, lst_facts in enumerate(sent2facts[sent]):
        # print(sent, lst_facts)
        data_to_write += "\t{}.{}. Facts extracted by Model {}:\n".format(i+1, alphab[j], j+1)
        #print(f"\t{i}.{alphab[j]}. Facts extracted by Model {j+1}:")
        data_to_write_csv.append(["\t{}.{}. Facts extracted by Model {}:\n".format(i+1, alphab[j], j+1) + str(lst_facts)])
        for fact in lst_facts:
            #print(f"\t\t- {fact}")
            data_to_write += "\t\t- {}\n".format(fact)
folder_path = f"~/FActScore/data/to_annotate_data/{lang}/task/sub_task_2/working_files/"
os.makedirs(folder_path, exist_ok=True)
txt_file_path = f"~/FActScore/data/to_annotate_data/{lang}/task/sub_task_2/working_files/sub-task-2-{annotator}.txt"

# Open the file in write mode
with open(txt_file_path, 'w') as file:
    # Write the data to the file
    file.write(data_to_write)
print(txt_file_path)
csv_file_path = f"~/FActScore/data/to_annotate_data/{lang}/task/sub_task_2/working_files/sub-task-2-{annotator}-answer-sheet.csv"
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)    
    writer.writerows(data_to_write_csv)

print(csv_file_path)
