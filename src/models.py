# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["ARK_API_KEY"] = ""
import time
import openai
from copy import deepcopy
import functools
from multiprocessing import Pool
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

try:
    from vllm import LLM, SamplingParams
except:
    print("Failed loading VLLM.")

openai_client = openai.AzureOpenAI(
    azure_endpoint="your_endpoint",
    api_version="your_api_version",
    api_key="your_api_key",
) 

def request_gpt41(messages: tuple, max_tokens: int, temperature: float, model: str = "", max_retries: int = 10, retry_delay: int = 10, output_logprob = True) -> list:
    i, messages = messages
    retries = 0
    while retries < max_retries:
        try:
            completion = openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                logprobs=True,
                top_logprobs=10,
            )

            output_logprobs = []
            logprobs = completion.choices[0].logprobs
            if logprobs and output_logprob:
                for idx in range(len(logprobs.content)):
                    logprob_dict = {
                        "token": logprobs.content[idx].token,
                        "top_logprobs": [{
                            "token": t.token,
                            "logprob": t.logprob,
                        } for t in logprobs.content[idx].top_logprobs],
                    }
                    output_logprobs.append(logprob_dict)

            return i, (completion.choices[0].message.content, output_logprobs)
        except Exception as e:
            retries += 1
            print(f"Attempt {retries} for request failed with error: {e}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
            if retries == max_retries:
                return i, ("ERROR", None)

class GPT41:
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict) -> list:
        request_new = functools.partial(
            request_gpt41, 
            max_tokens=param_dict["generation_params"]["max_new_tokens"], 
            temperature=param_dict["generation_params"]["temperature"],
            model="gpt-4.1-2025-04-14",
            max_retries=10,
            retry_delay=30,
            output_logprob=len(param_dict["care_tokens"]) > 0,
        )

        input_list = []
        for data_dict in data_list:
            data_dict = deepcopy(data_dict)
            input_list.append(data_dict)
        
        with Pool(8) as pool:
            results = pool.map(request_new, [
                (i, arg["messages"]) for i, arg in enumerate(input_list)])
        results = sorted(results, key=lambda x: x[0])
        final_results = [result for _, result in results]
        output_list = []
        for result in final_results:
            output_list.append({
                "output": result[0],
                "log_probs": result[1],
                "output_ids": [],
                "scores": []
            })
        return output_list

class Qwen2Model:
    def __init__(self, 
                 model_path, 
                 lora_path: str = None, 
                 use_flash_attention: bool = True, 
                 acceleration: str = "") -> None:
        self.model_path = model_path
        self.lora_path = lora_path
        self.use_flash_attention = use_flash_attention
        self.acceleration = acceleration
        
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        config.use_cache = True
        if use_flash_attention:
            config._attn_implementation = "flash_attention_2"
            
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
            use_fast=False,
            trust_remote_code=True,
        )

        if acceleration == "vllm":
            self.model = LLM(
                model=model_path,
                trust_remote_code=True,
                tensor_parallel_size=torch.cuda.device_count(),
                gpu_memory_utilization=0.90,
                max_seq_len_to_capture=8196,
                dtype="bfloat16",
            )
            if lora_path:
                raise ValueError("VLLM does not support lora branch, please merge lora into backbone first.")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                config=config, 
                torch_dtype="auto", 
                device_map="auto", 
            )
            if lora_path:
                print("loading lora ...")
                self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model = self.model.eval()

    def _infer_transformers(self, data_list: list, param_dict: dict = None) -> list:
        batch_data = []
        max_len = param_dict["max_length"]
        for data_dict in data_list:
            messages = data_dict["messages"]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                max_length=param_dict["max_length"],
                truncation=True,
            )
            max_len = min(max_len, len(input_ids))
            batch_data.append(input_ids)
        
        padded_ids = []
        padded_mask = []
        for i in range(len(batch_data)):
            if len(batch_data[i]) <= max_len:
                padded_ids.append([self.tokenizer.pad_token_id] * (max_len - len(batch_data[i])) + batch_data[i])
                padded_mask.append([0] * (max_len - len(batch_data[i])) + [1] * len(batch_data[i]))
            else:
                padded_ids.append(batch_data[i][- max_len : ])
                padded_mask.append([1] * max_len)
        padded_ids = torch.tensor(padded_ids, dtype=torch.int64).to("cuda")
        padded_mask = torch.tensor(padded_mask, dtype=torch.int32).to("cuda")

        with torch.no_grad():
            generated_dict = self.model.generate(
                padded_ids,
                attention_mask=padded_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                **param_dict["generation_params"],
            )
        
        sequences = generated_dict["sequences"]
        scores = generated_dict["scores"]
        output_list = []
        for batch_idx in range(len(padded_ids)):
            score_list = []
            for token_idx in range(len(scores)):
                score = scores[token_idx][batch_idx, param_dict["care_tokens"]].cpu().tolist()
                score_list.append(score)

            cur_output_ids = sequences[batch_idx, padded_ids.shape[1] : ].cpu().tolist()
            cur_output = self.tokenizer.decode(cur_output_ids, skip_special_tokens=True)
            
            output_list.append({
                "output": cur_output,
                "output_ids": cur_output_ids,
                "scores": score_list
            })
        
        return output_list

    def _infer_vllm(self, data_list: list, param_dict: dict) -> list:
        batch_data = []
        for data_dict in data_list:
            messages = data_dict["messages"]
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            batch_data.append(input_text)
        
        with torch.no_grad():
            outputs = self.model.generate(
                prompts=batch_data,
                sampling_params=SamplingParams(
                    top_p=param_dict["generation_params"]["top_p"],
                    top_k=param_dict["generation_params"]["top_k"],
                    truncate_prompt_tokens=param_dict["max_length"],
                    max_tokens=param_dict["generation_params"]["max_new_tokens"],
                ),
                use_tqdm=False,
            )
        
        output_list = []
        for batch_idx in range(len(batch_data)):
            if param_dict["care_tokens"]:
                score_list = [outputs[batch_idx].logits[0][batch_idx][param_dict["care_tokens"]].cpu().tolist()] 
            else:
                score_list = []
            output_list.append({
                "output": outputs[batch_idx].outputs[0].text,
                "output_ids": list(outputs[batch_idx].outputs[0].token_ids),
                "scores": score_list
            })
        
        return output_list
    
    @torch.inference_mode()
    def infer_batch(self, data_list: list, param_dict: dict) -> list:
        if self.acceleration == "vllm":
            return self._infer_vllm(data_list, param_dict)
        else:
            return self._infer_transformers(data_list, param_dict)