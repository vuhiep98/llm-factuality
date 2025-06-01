# compare resultant factscore between methods and provided labels 
import json
from tqdm import tqdm
model = "gpt4"
print(model)
data_path_01 = "~/FActScore/data/labeled/en_instances/" + model + "retrieval+bloomz_factscore_output_testing.json"
first_dict = {}
with open(data_path_01, "r") as f:
    for l in f:
        first_dict = json.loads(l)
print("----------------------")

provided_data_path = "~/FActScore/data/labeled/en_instances/" + model + ".jsonl"
provided_labels = []
how_many = 0
with open(provided_data_path) as f:
    for line in f:
        dp = json.loads(line) # dict_keys(['input', 'output', 'topic', 'cat', 'annotations'])
        #first_annotation = dp["annotations"][0] # dict_keys(['text', 'is-relevant', 'model-atomic-facts', 'human-atomic-facts'])
        texts = [e["text"] for e in dp["annotations"][0]["model-atomic-facts"]]
        
        llama_labels = [e=="S" for e in dp["annotations"][0]["LLAMA+NP_Labels"]]
        assert(len(texts) == len(llama_labels))
        if dp["annotations"][0]["ChatGPT_Labels"]:
            chatgpt_labels = [e=="S" for e in dp["annotations"][0]["ChatGPT_Labels"]]
            # print("LEN CHATGPT", dp)
            assert(len(texts) == len(chatgpt_labels))
        else:
            # print(dp["topic"])
            how_many += 1
            chatgpt_labels = None
        tmp = []
        for i, e in enumerate(texts):
            tmp.append((e, llama_labels[i], chatgpt_labels[i]) if chatgpt_labels is not None else (e, llama_labels[i], None))
        provided_labels.append(tmp)
true_llama = 0
true_chatgpt = 0
count_llama = 0
count_chatgpt = 0
chat_gpt_llama_agree = 0
has_disagreements = 0
llama_our_disagree = 0
chat_our_diagree = 0
our_true = 0
our_state_true = 0
our_state_true_chat_state_false = 0
our_state_true_llama_state_false = 0

# disparity = 0
# disparity_first_true = 0
for i in range(len(first_dict["decisions"])):
    first_instance = first_dict["decisions"][i]
    second = provided_labels[i]
    for j in range(len(first_instance)):
        count_llama += 1
        if first_instance[j]["is_supported"] == second[j][1]:
            true_llama += 1
        if second[j][2] is not None:
            count_chatgpt += 1
            if first_instance[j]["is_supported"] == second[j][2]:
                true_chatgpt += 1
            if second[j][1] == second[j][2]:
                chat_gpt_llama_agree += 1
            if first_instance[j]["is_supported"]:
                our_state_true += 1
                if not second[j][1]:
                    our_state_true_llama_state_false += 1
                if not second[j][2]:
                    our_state_true_chat_state_false += 1
            if first_instance[j]["is_supported"] != second[j][1] or first_instance[j]["is_supported"] != second[j][2] or second[j][2] != second[j][1]:
                has_disagreements += 1
                if first_instance[j]["is_supported"] != second[j][1]:
                    if first_instance[j]["is_supported"]:
                        our_true += 1
                    else:
                        print(first_instance[j], "#our:", first_instance[j]["is_supported"],"#llama:", second[j][1], "#chatgpt:", second[j][2])
                    llama_our_disagree += 1
                if first_instance[j]["is_supported"] != second[j][2]:
                    chat_our_diagree += 1
                
print("LLAMA percentage:", true_llama/count_llama, true_llama, count_llama)
print("CHATGPT percentage:", true_chatgpt/count_chatgpt, true_chatgpt, count_chatgpt)
print("LLAMA/CHATGPT percentage:", chat_gpt_llama_agree/count_chatgpt, chat_gpt_llama_agree, count_chatgpt)
print("llama_our_disagree", llama_our_disagree/has_disagreements, our_true/llama_our_disagree, llama_our_disagree, has_disagreements)
print("chat_our_diagree", chat_our_diagree/has_disagreements, chat_our_diagree, has_disagreements)
print("our_state_true_llama_state_false", our_state_true_llama_state_false/our_state_true)
print("our_state_true_chat_state_false", our_state_true_chat_state_false/our_state_true)