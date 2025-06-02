# compare resultant factscore between methods 
import json
lang = "es"
model = "gpt4"
evaluator = "mistral"
data_path_01 = f"~/FActScore/data/to_evaluate/{lang}/sub_gpt4/en_instances/{model}en_retrieval+GPT-4_factscore_output_provided_facts.json"
data_path_02 = f"~/FActScore/data/to_evaluate/{lang}/sub_gpt4/{model}{lang}_retrieval+GPT-4_factscore_output_provided_facts.json"


# data_path_01 = f"~/FActScore/data/to_evaluate/{lang}/en_instances/{model}en_retrieval+{evaluator}_factscore_output_provided_facts.json"
# data_path_02 = f"~/FActScore/data/to_evaluate/{lang}/{model}{lang}_retrieval+{evaluator}_factscore_output_provided_facts.json"

# data_path_01 = f"~/FActScore/data/to_evaluate/{lang}/en_instances/{model}en_retrieval+mistral_factscore_output_provided_facts.json"
# data_path_02 = f"~/FActScore/data/to_evaluate/{lang}/{model}{lang}_retrieval+GPT-4_factscore_output_provided_facts.json"
# data_path_01 = f"~/FActScore/data/to_evaluate/{lang}/{model}{lang}_retrieval+mistral_factscore_output_provided_facts.json"
# data_path_02 = f"~/FActScore/data/to_evaluate/{lang}/{model}{lang}_retrieval+GPT-4_factscore_output_provided_facts.json"
lang_1 = "ar"
lang_2 = "bn"
data_path_01 = f"~/translate_data/en_instances/{lang_1}_instances/gpt4_sub{lang_1}_retrieval+GPT-4_factscore_output_provided_facts.json"
data_path_02 = f"~/translate_data/en_instances/{lang_2}_instances/gpt4_sub{lang_2}_retrieval+GPT-4_factscore_output_provided_facts.json"
first_dict = {}
second_dict = {}
print(data_path_01)
print(data_path_02)
with open(data_path_01, "r") as f:
    for l in f:
        first_dict = json.loads(l)
with open(data_path_02, "r") as f:
    for l in f:
        second_dict = json.loads(l)
print(len(first_dict["decisions"]))
print("----------------------")
print(len(second_dict["decisions"]))
true = 0
count = 0
disparity = 0
disparity_first_true = 0
tt = 0
ff = 0
ft = 0
tf = 0
for i in range(len(first_dict["decisions"])):
    first_instance = first_dict["decisions"][i]
    second_instance = second_dict["decisions"][i]
    # print(first_instance[0], second_instance[0])
    if len(first_instance) != len(second_instance):
        print(first_instance[0], len(first_instance), second_instance[0], len(second_instance))
        continue
    for j in range(len(first_instance)):
        
        if first_instance[j]["is_supported"] and second_instance[j]["is_supported"]:
            tt += 1
        elif first_instance[j]["is_supported"] and not second_instance[j]["is_supported"]:
            tf += 1
        elif not first_instance[j]["is_supported"] and second_instance[j]["is_supported"]:
            ft += 1
        elif not first_instance[j]["is_supported"] and not second_instance[j]["is_supported"]:
            ff += 1
all = tt+ft+tf+ff
po = (tt+ff)/all
pe = ((tf+tt)/all) * ((ft+ff)/all)

k = (po - pe) / (1 - pe)
print("Kappa Score:", k)
print("tt:", tt, "tf:", tf, "ft:", ft, "ff:", ff)
print("Agreement:", po)