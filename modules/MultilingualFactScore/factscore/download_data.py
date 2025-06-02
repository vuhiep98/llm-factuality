import argparse
import os
import subprocess
import torch
import tqdm
import transformers
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
def download_file(_id, dest, cache_dir):
    if os.path.exists(dest) or os.path.exists(os.path.join(cache_dir, dest)):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if os.path.exists(dest.replace(".zip", "")) or os.path.exists(os.path.join(cache_dir, dest.replace(".zip", ""))):
        print ("[Already exists] Skipping", dest)
        print ("If you want to download the file in another location, please specify a different path")
        return

    if "/" in dest:
        dest_dir = "/".join(dest.split("/")[:-1])
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
    else:
        dest_dir = "."

    if _id.startswith("https://"):
        command = """wget -O %s %s""" % (dest, _id)
    else:
        command = """wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=%s' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id=%s" -O %s && rm -rf /tmp/cookies.txt""" % (_id, _id, dest)

    ret_code = subprocess.run([command], shell=True)
    if ret_code.returncode != 0:
        print("Download {} ... [Failed]".format(dest))
    else:
        print("Download {} ... [Success]".format(dest))

    if dest.endswith(".zip"):
        command = """unzip %s -d %s && rm %s""" % (dest, dest_dir, dest)

        ret_code = subprocess.run([command], shell=True)
        if ret_code.returncode != 0:
            print("Unzip {} ... [Failed]".format(dest))
        else:
            print("Unzip {} ... [Success]".format(dest))



def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def recover_instruct_llama(path_raw, output_path, device="cpu", test_recovered_model=False):
    """Heavily adapted from https://github.com/tatsu-lab/stanford_alpaca/blob/main/weight_diff.py."""

    # model_raw = transformers.AutoModelForCausalLM.from_pretrained(
    #     path_raw,
    #     device_map={"": torch.device(device)},
    #     torch_dtype=torch.float32,
    #     low_cpu_mem_usage=True,
    #     access_token = "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"
    # )
    # model_recovered = transformers.AutoModelForCausalLM.from_pretrained(
    #     "kalpeshk2011/instruct-llama-7b-wdiff",
    #     device_map={"": torch.device(device)},
    #     torch_dtype=torch.float32,
    #     low_cpu_mem_usage=True,
    #     access_token = "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"
    # )

    # tokenizer_raw = transformers.AutoTokenizer.from_pretrained(path_raw)
    # if tokenizer_raw.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token="[PAD]"),
    #         model=model_raw,
    #         tokenizer=tokenizer_raw,
    #     )
    # tokenizer_recovered = transformers.AutoTokenizer.from_pretrained("kalpeshk2011/instruct-llama-7b-wdiff")

    # state_dict_recovered = model_recovered.state_dict()
    # state_dict_raw = model_raw.state_dict()
    # for key in tqdm.tqdm(state_dict_recovered):
    #     state_dict_recovered[key].add_(state_dict_raw[key])
    model_recovered = transformers.AutoModelForCausalLM.from_pretrained(
        path_raw,
        device_map={"": torch.device(device)},
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    # model_recovered = transformers.AutoModelForSeq2SeqLM.from_pretrained(
    #     path_raw,
    #     device_map={"": torch.device(device)},
    #     torch_dtype=torch.float32,
    #     low_cpu_mem_usage=True,
    # )
    # model_recovered.resize_token_embeddings(260164)
    
    tokenizer_recovered = transformers.AutoTokenizer.from_pretrained(path_raw)
    # if tokenizer_recovered.pad_token is None:
    #     smart_tokenizer_and_embedding_resize(
    #         special_tokens_dict=dict(pad_token="[PAD]"),
    #         model=model_recovered,
    #         tokenizer=tokenizer_recovered)
    # model_recovered = PeftModel.from_pretrained(model_recovered, "MaLA-LM/mala-500")

    # if test_recovered_model:
    #     # input_text = (
    #     #     "Below is an instruction that describes a task. "
    #     #     "Write a response that appropriately completes the request.\r\n\r\n"
    #     #     "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
    #     # )
    #     input_text = (
    #         "Below is an instruction that describes a task. "
    #         "Write a response that appropriately completes the request.\r\n\r\n"
    #         "### Instruction:\r\nList three technologies that make life easier.\r\n\r\n### Response:"
    #     )
    #     inputs = tokenizer_recovered(input_text, return_tensors="pt")
    #     out = model_recovered.generate(inputs=inputs.input_ids, max_new_tokens=1000)
    #     output_text = tokenizer_recovered.batch_decode(out, skip_special_tokens=True)[0]
    #     output_text = output_text[len(input_text) :]
    #     print(f"Input: {input_text}\nCompletion: {output_text}")
    if output_path is not None:
        print("START SAVING")
        # model_recovered.save_pretrained(output_path, save_embedding_layers=False)
        model_recovered.save_pretrained(output_path)
        tokenizer_recovered.save_pretrained(output_path)
    return model_recovered, tokenizer_recovered

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore")
    parser.add_argument('--model_HF_path',
                        type=str,
                        default=None)

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)
    print(args.model_dir)

    # Comment out after downloading data for the first time
    download_file("1IseEAflk1qqV0z64eM60Fs3dTgnbgiyt", "demos.zip", args.data_dir)
    download_file("1enz1PxwxeMr4FRF9dtpCPXaZQCBejuVF", "data.zip", args.data_dir)

    print(args.model_HF_path.split("/")[-1])
    # inst-Mistral-7B-Instruct-v0.2
    if args.model_HF_path:
        recover_instruct_llama(args.model_HF_path, os.path.join(args.model_dir, "inst-{}-Instruct".format(args.model_HF_path.split("/")[-1])), test_recovered_model= True)

    # download the roberta_stopwords.txt file; comment out after the first download
    # subprocess.run(["wget https://raw.githubusercontent.com/shmsw25/FActScore/main/roberta_stopwords.txt"], shell=True)

    # move the files to the data directory; comment out after the first download
    subprocess.run(["mv demos %s" % args.data_dir], shell=True)

