import argparse
import string
import json
import numpy as np
import os
import logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval
from factscore.gemini_lm import GeminiModel

langs_avai = {"es": 5, "bn": 8, "ar":8, "en":5, "fr": 5, "vi":5, "zh-cn":5, "ru":5, "hi":8}

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 evaluated_model = "",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 lang="en",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256):
        assert model_name in ["mistral", "ChatGPT", "GPT-4", "Gemini-Pro","retrieval+mistral","retrieval+bloomz", "retrieval+bloomz+npm", "retrieval+llama", "retrieval+llama+npm", "retrieval+mistral+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm", "retrieval+GPT-4", "retrieval+Gemini-Pro", "retrieval+GPT-4+npm"]
        self.model_name = model_name

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type
        self.lang = lang
        self.data_dir = data_dir
        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.evaluated_model = evaluated_model
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "bloomz" in model_name:
            self.lm = CLM("inst-bloomz-7b1",
                          model_dir=os.path.join(model_dir, "inst-bloomz-7b1"),
                          cache_file=os.path.join(cache_dir, f"inst-bloomz-7b1-{lang}-{evaluated_model}.pkl")) # change cache of bloomz frequently
        elif "mistral" in model_name:
            self.lm = CLM("inst-Mistral-7B-Instruct-v0.2",
                          model_dir=os.path.join(model_dir, "inst-Mistral-7B-Instruct-v0.2"),
                          cache_file=os.path.join(cache_dir, "inst-Mistral-7B-Instruct-v0.2-"+lang+".pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT-evaluator.pkl"),
                                  key_path=openai_key)
        elif "GPT-4" in model_name:
            self.lm = OpenAIModel("ChatGPT-gpt4",
                                  cache_file=os.path.join(cache_dir, "GPT-4-evaluator.pkl"),
                                  key_path=openai_key)
        elif "Gemini-Pro" in model_name:
            self.lm = gemini_lm = GeminiModel("Gemini-Pro", cache_file=os.path.join(cache_dir, "Gemini-pro-evaluator.pkl"), key_path="~/projects/open_ai_key.txt")
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval_n_8-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval_n_8-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            # print("YESSSSSSSSSSSS NPM")
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.0005
        elif model == "gpt-3.5-turbo":
            rate = 0.0005
        elif model == "gpt-4-turbo":
            rate = 0.03

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.2f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"
        sent2facts = []
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        model_dir=self.model_dir,
                                                        cache_dir=self.cache_dir,
                                                        lang=self.lang,
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "GPT3.5-turbo(1).pkl"))

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="gpt-3.5-turbo")

            if verbose:
                topics = tqdm(topics)

            atomic_facts = []
            
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                sent2fact = curr_afs.copy()
                sent2facts.append(sent2fact)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")
        if "GPT-4" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-4-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                score = np.mean([d["is_supported"] for d in decision])
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {
               "model": self.model_name,
               "sent2facts": sent2facts,
               "score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                if "retrieval" in self.model_name:
                    passages = self.retrieval[knowledge_source].get_passages(topic, atom, 8)
                    if passages == None or len(passages) == 0:
                        decisions.append({"atom": atom, "is_supported": bool(False)})
                        continue
                    if "bloomz" in self.model_name:
                        definition = "Answer the question about {} based on the given context.\n\nText: ".format(topic)
                        context = ""
                        # for psg_idx, psg in enumerate(reversed(passages)):
                        #     context += "{}\n".format(psg["text"].replace("<s>", "").replace("</s>", ""))
                        definition += context.strip()        
                        if not definition[-1] in string.punctuation:
                            definition += "."
                        prompt = "{}\n\nQuestion: {} True or False? \nAnswer: ".format(definition.strip(), atom.strip())
                        
                    else:
                        definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                        context = ""
                        for psg_idx, psg in enumerate(reversed(passages)):
                            context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                        # psg_all = "".join([psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages])
                        # context += "Title: {}\nText: {}\n\n".format(passages[0]["title"], psg_all)
                        definition += context.strip()
                        if not definition[-1] in string.punctuation:
                            definition += "."
                        prompt = "{}\n\nInput: {} True or False? \nOutput: ".format(definition.strip(), atom.strip())
                        print("LENGTH PROMPT:", len(prompt))
                else:
                    definition = "Answer the question about {}.".format(topic)
                    prompt = "{}\nInput: {} True or False? \nOutput: ".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue
                is_supported = True
                output = self.lm.generate(prompt)
                
                print("CONTEXT", prompt)
                print("OUTPUT", output)
                # explain_prompt = prompt + output[0][1:-4] + ".\nWhy? Explanation:"
                # print("EXPLAIN_CONTEXT", explain_prompt)
                # output2 = self.lm.generate(explain_prompt)
                # print("EXPLAIN", output2)
                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    if "bloomz" in self.model_name:
                        true_score = logits[22096]
                        false_score = logits[27092]
                    elif "llama" in self.model_name:
                        true_score = logits[5852]
                        false_score = logits[7700]
                    elif "mistral" in self.model_name:
                        true_score = logits[6110]
                        false_score = logits[8250]
                    is_supported = true_score > false_score
                    print(is_supported, true_score, false_score)
                else:
                    # when logits are unavailable
                    if "Gemini-Pro" in self.model_name:
                        output = (output, 0)
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True
            
            print("----------------------------------------------------")
            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                print("npprob", npprob)
                is_supported = npprob > 0.3

            decisions.append({"atom": atom, "is_supported": bool(is_supported)})

        if cost_estimate:
            return total_words
        else:
            return decisions

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)
    parser.add_argument('--lang',
                        type=str,
                        default="en")
    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()
    print(args)
    logging.critical(args)
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    evaluated_model = args.input_path.split("/")[-1].split(".")[0],
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    lang=args.lang,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    if args.n_samples is not None and tot==args.n_samples:
                        break 
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                lst_facts = []
                for sent in dp["annotations"]:
                    for atom in sent["model-atomic-facts"]:
                        if type(atom) == dict:
                            assert "text" in list(atom.keys())
                            lst_facts.append(atom["text"])
                        elif type(atom) == str:
                            lst_facts.append(atom)
                        else:
                            assert False
                atomic_facts.append(lst_facts)
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))
    # print("OUTTTTTTTTTT",out)
    # Save out as a json file

    if args.use_atomic_facts:
        with open(args.input_path.replace(".jsonl", f"{args.lang}_"+args.model_name+f"_factscore_output_provided_facts_put_all.json"), 'w') as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        with open(args.input_path.replace(".jsonl",f"{args.lang}_"+args.model_name+f"_factscore_output_gen_facts.json"), 'w') as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

