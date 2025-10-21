# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import time
from copy import deepcopy
from openai import AzureOpenAI
 
class Client_GPT4d1():
    def __init__(self, ak="your_api_key", model="gpt-4.1-2025-04-14", max_tokens=8192, system=None):
        self.system = system
        self.client = AzureOpenAI(
            api_key = ak,
            api_version = "your_api_version",
            azure_endpoint = "your_endpoint",
        )

        self.payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": ''
                }
            ],
            "max_tokens": max_tokens
        }
        if self.system:
            self.payload['messages'].insert(0, {"role": "system", "content": self.system})
        
    
    def __call__(self, input_text):
        prompt = [{
                    "type": "text",
                    "text": input_text
                }]
        payload = deepcopy(self.payload)
        if self.system:
            payload['messages'][1]['content'] = prompt
        else:
            payload['messages'][0]['content'] = prompt

        for i in range(1, 3):
            try:
                response = self.client.chat.completions.create(**payload, timeout=100)
                response = response.choices[0].message.content
                if response == '': raise Exception('blank response')
                return response
            except Exception as e:
                response  = f'error: {e}'
                print(response)
                time.sleep(i*10)
        return response
