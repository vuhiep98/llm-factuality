import argparse
import string
import json
import numpy as np
import os
import logging
from transformers import logging as tfm_logging

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

tfm_logging.set_verbosity_error()

class FactScorer(object):

    def __init__(self,
                 model_name="ChatGPT+retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256,
                 use_cache=True):
        
        self.model_name = model_name
        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate
        self.use_cache = use_cache

        llm_model_name = self.model_name.split("+")[-1]
        if llm_model_name == "llama3":
            self.lm = CLM(model_name="Llama-3.1-8B-Instruct",
                          model_dir=self.model_dir,
                          cache_file=os.path.join(self.cache_dir, "llama3.1-8B-Instruct.pkl"),
                          use_cache=self.use_cache)
        elif llm_model_name == "llama2":
            self.lm = CLM(model_name="inst-llama-7B",
                          model_dir=self.model_dir,
                          cache_file=os.path.join(self.cache_dir, "inst-llama-7B.pkl"),
                          use_cache=self.use_cache)
        elif llm_model_name == "ChatGPT":
            self.lm = OpenAIModel("ChatGPT", 
                                cache_file=os.path.join(self.cache_dir, "GPT4.pkl"), 
                                key_path=openai_key,)
        else:
            print("None LLM is loaded")
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

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
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
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.002

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

        af_gen_name = self.model_name.split("+")[0]
        
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
            # Some model atomic facts are empty, re-generate them
            for i, (facts, gen) in enumerate(zip(atomic_facts, generations)):
                if len(facts) == 0:
                    if self.af_generator is None:
                        af_gen_name = self.model_name.split("+")[0]
                        if af_gen_name == "ChatGPT":
                            self.af_generator = AtomicFactGenerator(
                                key_path=self.openai_key,
                                demon_dir=os.path.join(self.data_dir, "demos"),
                                cache_file=os.path.join(self.cache_dir, f"AT_InstructGPT.pkl")
                            )
                        elif af_gen_name == "llama2":
                            self.af_generator = AtomicFactGenerator(
                                demon_dir=os.path.join(self.data_dir, "demos"), 
                                model_name="llama2",
                                cache_file=os.path.join(self.cache_dir, "af-inst-llama-7B.pkl")
                            )
                        elif af_gen_name == "llama3":
                            self.af_generator = AtomicFactGenerator(
                                demon_dir=os.path.join(self.data_dir, "demos"), 
                                model_name="llama3",
                                cache_file=os.path.join(self.cache_dir, "af-llama3.1-8B-Instruct.pkl")
                            )
                    
                    response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                    if response_abstained:
                        atomic_facts.append(None)
                        continue
                    curr_afs, _ = self.af_generator.run(gen)
                    curr_afs = [fact for _, facts in curr_afs for fact in facts]
                    atomic_facts[i] = curr_afs
                    self.af_generator.save_cache()
        else:
            if self.af_generator is None:
                if af_gen_name == "ChatGPT":
                    self.af_generator = AtomicFactGenerator(
                        key_path=self.openai_key,
                        demon_dir=os.path.join(self.data_dir, "demos"),
                        cache_file=os.path.join(self.cache_dir, "AT_InstructGPT.pkl")
                    )
                elif af_gen_name == "llama2":
                    self.af_generator = AtomicFactGenerator(
                        demon_dir=os.path.join(self.data_dir, "demos"), 
                        model_name="llama2",
                        cache_file=os.path.join(self.cache_dir, "af-inst-llama-7B.pkl")
                    )
                elif af_gen_name == "llama3":
                    self.af_generator = AtomicFactGenerator(
                        demon_dir=os.path.join(self.data_dir, "demos"), 
                        model_name="llama3",
                        cache_file=os.path.join(self.cache_dir, "af-llama3.1-8B-Instruct.pkl")
                    )
            
            if af_gen_name == "ChatGPT":
                # estimate the total cost of atomic fact generation
                total_words = 0
                for gen in generations:
                    total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

                self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

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
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append([])
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 10 == 0:
                    self.af_generator.save_cache()

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        verificator = self.model_name.split("+")[-1]
        if verificator == "ChatGPT":
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None or len(facts) == 0:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                score = np.mean([d["is_supported"] for d in decision]) if len(decision) > 0 else 0.0
                
                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                if len(scores) % 10 == 0:
                    self.save_cache()

        self.save_cache()

        out = {"score": np.mean(scores),
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)
        
        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        total_words = 0
        
        if len(atomic_facts) == 0:
            # optionally, first detect if the response is abstained
            response_abstained = is_response_abstained(generation, self.abstain_detection_type)
            if not response_abstained:
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(generation)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs) > 0:
                    atomic_facts = curr_afs
                self.af_generator.save_cache()
                
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())

                if cost_estimate:
                    if self.use_cache and cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue
                
                output = self.lm.generate(prompt)
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

            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
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
                        default="ChatGPT+retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")
    parser.add_argument('--output_path',
                        type=str,
                        default="outputs/labeled/InstructGPT_factscore.json",
                        help="output path to save the results")
    parser.add_argument('--use_cache',
                        action="store_true",
                        help="whether to use cache for the model and retrieval")

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

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type,
                    use_cache=args.use_cache)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                topics.append(dp["topic"])
                generations.append(dp["output"])
                facts = []
                if "annotations" in dp and dp["annotations"]:
                    for sent in dp["annotations"]:
                        if "model-atomic-facts" in sent and sent["model-atomic-facts"]:
                            facts += [atom["text"] for atom in sent["model-atomic-facts"]]
                        else:
                            facts += []
                atomic_facts.append(facts)
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

    # Save out as a json file
    llm_name = args.input_path.split("/")[-1].split(".")[0]
    output_path = os.path.join(args.output_path, args.model_name)
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"{llm_name}_{args.model_name}.json")
    json.dump(out, open(output_file, "w"), indent=4)

