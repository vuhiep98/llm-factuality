from factscore.lm import LM
import sys
import time
import os
import numpy as np
import logging
import google.generativeai as genai
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

class GeminiModel(LM):

    def __init__(self, model_name, cache_file=None, key_path=""):
        self.model_name = model_name
        self.key_path = key_path
        self.generation_config =  {
            "temperature": 0.7,
        }
        self.save_interval = 100
        super().__init__(cache_file)

    def load_model(self):
        # load api key
        key_path = self.key_path
        genai.configure(api_key="")
        self.model = genai.GenerativeModel('gemini-pro', safety_settings=safety_settings, generation_config=self.generation_config)
    def _generate(self, prompt, max_sequence_length=2048, max_output_length=128):
        if self.add_n % self.save_interval == 0:
            self.save_cache()
        # return a tuple of string (generated text) and metadata (any format)
        # This should be about generating a response from the prompt, no matter what the application is
        response = None
        received = False
        num_rate_errors = 0
        while not received:
            try:
                response = self.model.generate_content(prompt).text
                received = True
            except:
                error = sys.exc_info()[0]
                num_rate_errors += 1
                print(error)
                # genai.
                # if error == genai.error.InvalidRequestError:
                #     # something is wrong: e.g. prompt too long
                #     logging.critical(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                #     assert False
                logging.error("API error: %s (%d)" % (error, num_rate_errors))
                time.sleep(np.power(2, num_rate_errors))
                if num_rate_errors > 5:
                    return "False"
                    assert False
        # time.sleep(5)
        # output = self.model.generate_content(prompt).text
        return response