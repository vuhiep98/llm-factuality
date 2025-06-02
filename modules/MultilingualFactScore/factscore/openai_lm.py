from factscore.lm import LM
import openai
import sys
import time
import os
import numpy as np
import logging

class OpenAIModel(LM):

    def __init__(self, model_name, cache_file=None, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.7
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        assert os.path.exists(key_path), f"Please place your OpenAI APT Key in {key_path}."
        with open(key_path, 'r') as f:
            api_key = f.readline()
        # openai.api_key = api_key.strip()
        if "gpt4" in self.model_name:
            openai.api_type = "azure"
            openai.api_base = ""
            openai.api_key = ""
            openai.api_version = ""
            print("DONE SET UP, GPT-4")
        elif self.model_name == "ChatGPT":
            openai.api_type = "azure"
            openai.api_base = ""
            openai.api_key = ""
            openai.api_version = ""
            print("DONE SET UP, ChatGPT")
        # openai.api_key = api_key.strip()
        # openai.api_version = ""
        self.model = self.model_name

    def _generate(self, prompt, max_sequence_length=4096, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        if self.model_name == "ChatGPT":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            try:
                output = response["choices"][0]["message"]["content"]
            except Exception as error:
                print("message:", message)
                print("_generate Error:", error)
                print("response:", response)
                #assert False, "Stop any way"
                output = "InvalidRequestionError: The response was filtered due to the prompt triggering Azure OpenAI's content management policy. We apologize for not being able to process your request."                
            return output, response
        elif self.model_name == "InstructGPT":
            # Call API
            response = call_GPT3(prompt, temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        elif self.model_name == "ChatGPT-gpt4":
            # Construct the prompt send to ChatGPT
            message = [{"role": "user", "content": prompt}]
            # Call API
            response = call_ChatGPT(message, model_name="gpt-4-0125", temp=self.temp, max_len=max_sequence_length)
            # Get the output from the response
            try:
                output = response["choices"][0]["message"]["content"]
            except Exception as error:
                print("Error:", error)
                print("Response:", response)
                logging.critical(f"Error: {error}")
                logging.critical(f"Response: {response}")
                output = ""
                
            return output, response
        elif self.model_name == "ChatGPT-gpt4-fs":
            # Construct the prompt send to ChatGPT
            # Call API
            response = call_GPT3(prompt, model_name="gpt-4-0125", temp=self.temp)
            # Get the output from the response
            output = response["choices"][0]["text"]
            return output, response
        else:
            print(self.model_name)
            raise NotImplementedError()

def call_ChatGPT(message, model_name="gpt35-1106", max_len=1024, temp=0.7, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            if model_name=="gpt-4-0125":
                time.sleep(15)
            else:
                time.sleep(2)
            response = openai.ChatCompletion.create(engine=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
            received = True
        except Exception as error:
            print("Error:", error)
            num_rate_errors += 1
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                tmp_dict = {}
                tmp_dict["choices"] = [{"message": {"content": "InvalidRequestionError: The response was filtered due to the prompt triggering Azure OpenAI's content management policy. We apologize for not being able to process your request"}}]
                return tmp_dict
            
            logging.error("API error: %s (%d). Waiting %dsec" % (error, num_rate_errors, np.power(2, num_rate_errors)))
            time.sleep(np.power(2, num_rate_errors))
    return response


def call_GPT3(prompt, model_name="text-davinci-003", max_len=512, temp=0.7, num_log_probs=0, echo=False, verbose=False):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            response = openai.Completion.create(engine=model_name,
                                                prompt=prompt,
                                                max_tokens=max_len,
                                                temperature=temp,
                                                logprobs=num_log_probs,
                                                echo=echo)
            received = True
        except:
            error = sys.exc_info()[0]
            num_rate_errors += 1
            if error == openai.error.InvalidRequestError:
                # something is wrong: e.g. prompt too long
                logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            logging.error("API error: %s (%d)" % (error, num_rate_errors))
            time.sleep(np.power(2, num_rate_errors))
    return response
