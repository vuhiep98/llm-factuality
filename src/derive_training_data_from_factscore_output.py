import json
import re
import sys
import spacy
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from pathlib import Path
from nltk.tokenize import sent_tokenize
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

bert_tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")
bert_model = BertModel.from_pretrained("google-bert/bert-base-uncased", device_map="cuda")

nlp = spacy.load("en_core_web_sm")
ners = ["PERSON", "GPE", "ORG", "PRODUCT"]

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def format_ne(text):
    ne_map = {}
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ners and "." in ent.text:
            formatted_ne = ent.text.replace(".", " ")
            formatted_ne = " ".join([t.strip() for t in formatted_ne.split()])
            text = text.replace(ent.text, formatted_ne)
            ne_map[formatted_ne] = ent.text
    return text, ne_map

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def split_sentences(lm_generation):
    ne_formatted_text, ner_maps = format_ne(lm_generation["output"])
    sentences = sent_tokenize(ne_formatted_text)
    initials = detect_initials(lm_generation["input"])
    sentences = fix_sentence_splitter(sentences, initials)
    for i, sentence in enumerate(sentences):
        for formatted_ne, original_ne in ner_maps.items():
            sentence = sentence.replace(formatted_ne, original_ne)
        sentences[i] = sentence
    return sentences

def extract_facts_to_list(input_text):
    lines = input_text.strip().split('\n')
    facts_list = []
    for line in lines:
        line = line.strip()
        if line.startswith('- '):
            facts_list.append(line[2:])
    return facts_list

############ Similarity
def ngram_jaccard_similarity(str1, str2, n=2):
    def generate_ngrams(string, n):
        return {" ".join(string[i:i+n]) for i in range(len(string) - n + 1)}
    ngrams1 = generate_ngrams(str1.lower().split(), n)
    ngrams2 = generate_ngrams(str2.lower().split(), n)

    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)

    return len(intersection) / len(union) if union else 0.0

def jaccard_similarity(str1, str2):
    unigram_score = ngram_jaccard_similarity(str1, str2, n=1)
    bigram_score = ngram_jaccard_similarity(str1, str2, n=2)
    trigram_score = ngram_jaccard_similarity(str1, str2, n=3)
    score = np.mean([unigram_score, bigram_score, trigram_score])
    return score

def bert_similarity(str1, str2):
    input1 = bert_tokenizer(str1, return_tensors="pt").to(bert_model.device)
    input2 = bert_tokenizer(str2, return_tensors="pt").to(bert_model.device)
    emb1 = bert_model(**input1).last_hidden_state[0][0].cpu().detach().numpy().reshape((1, 768))
    emb2 = bert_model(**input2).last_hidden_state[0][0].cpu().detach().numpy().reshape((1, 768))
    return cosine_similarity(emb1, emb2)[0][0]

def text_similarity(str1, str2):
    jaccard_score = jaccard_similarity(str1, str2)
    bert_score = bert_similarity(str1, str2)
    similarity_score = 0.75*jaccard_score + 0.25*bert_score
    return similarity_score
#########################

def get_atomic_facts(sentences, decisions, threshold=0.5):
    sentence2facts = {}
    for sentence in sentences:
        sentence2facts.setdefault(sentence, [])
    
    if decisions is not None:
        for fact in decisions:
            scores = [(i, text_similarity(sentence, fact["atom"])) for i, sentence in enumerate(sentences)]
            best_score = max(scores, key=lambda x: x[1])
            if best_score[1] > threshold:
                sentence2facts[sentences[best_score[0]]].append(fact)
    
    return sentence2facts

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, required=True)
    parser.add_argument("--threshold", type=float, required=True, default=0.5)
    args = parser.parse_args()
    
    llm_unlabled_folder = Path(__file__).parents[1]/"modules"/"FActSCORE"/"factscore_data"/"unlabeled"
    llm_output_file = llm_unlabled_folder/f"{args.llm}_factscore_output.json"
    llm_input_file = llm_unlabled_folder/f"{args.llm}.jsonl"
    llm_mapped_file = llm_unlabled_folder/f"{args.llm}_factscore_output_mapped.jsonl"
    threshold = args.threshold
    if threshold > 1 or threshold < 0:
        print("Threshold must be between 0 and 1")
        sys.exit(1)
    
    factscores = json.load(open(llm_output_file))
    inputs = read_jsonl(llm_input_file)

    input_count = 0
    factscore_count = 0
    
    with tqdm(total=len(inputs)) as pbar:
        while input_count < len(inputs):
            
            if 'Francisco Urroz' in inputs[input_count]["input"]:
                input_count += 1
                pbar.update(1)
                continue
                
            sentences = split_sentences(inputs[input_count])
            decisions = factscores["decisions"][factscore_count]
            sentence2facts = get_atomic_facts(sentences, decisions, threshold)
            inputs[input_count]['factscores'] = sentence2facts
            
            input_count += 1
            factscore_count += 1
            pbar.update(1)

    # Write to file:
    print(f"Dump to {llm_mapped_file}")
    with open(args.predict_file, "w") as f:
        json.dump(inputs, f, indent=2)