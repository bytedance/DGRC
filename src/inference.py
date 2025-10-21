# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
from tqdm import tqdm
import argparse
from copy import deepcopy
from src.models import GPT41, Qwen2Model

PARAMS = {
    "train_set/MedMCQA": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "train_set/CaseHOLD": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "test_set/MedMCQA": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "test_set/MedQA": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "test_set/MMLU-M": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "test_set/CaseHOLD": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
    "test_set/MMLU-L": {
        "max_length": 4096,
        "care_tokens": [],  
        "generation_params": {
            "max_new_tokens": 4096,
            "do_sample": False, 
            "top_p": 0.95, 
            "top_k": 30, 
            "num_beams": 1,
            'temperature': 0.6,
        },
    },
}

class Infer:
    def __init__(self, 
        model_path, 
        lora_path, 
        use_flash_attention: bool = True, 
        acceleration: str = "", 
        model_class: str = "Qwen2Model",
        task_list: str = "",
        batch_size: int = 1, 
        data_path: str = "data", 
        save_path: str = "results", 
        save_id: str = "example",
        samples: int = 8) -> None:

        self.batch_size = batch_size
        self.all_param_dict = PARAMS
        self.data_path = data_path
        self.save_path = save_path
        self.save_id = save_id
        self.samples = samples
        
        self.model = eval(model_class)(
            model_path, 
            lora_path, 
            use_flash_attention, 
            acceleration,
        )
        self.task_list = task_list.split(",")
        self.data_map = dict()
        self._load_data()
    
    def _load_data(self):
        task_domain_list = os.listdir(self.data_path)
        for task_domain in task_domain_list:
            for task_name in os.listdir(os.path.join(self.data_path, task_domain)):
                task_path = os.path.join(self.data_path, task_domain, task_name)
                task_key = "{}/{}".format(
                    task_domain,
                    task_name
                )

                if "ALL" in self.task_list or task_key in self.task_list:
                    self.data_map[task_key] = [
                        os.path.join(task_path, fn) for fn in os.listdir(task_path) if fn.endswith(".json")
                    ]
        
        print("DATA_INFO: {}".format(
            json.dumps(self.data_map, ensure_ascii=False, indent=2)
        ))
    
    def _convert_data(self, batch_data):
        output_data = []
        for data_dict in batch_data:
            conversations = data_dict["conversations"]
            messages = []

            if "system_prompt" in data_dict:
                messages.append({
                    "role": "system",
                    "content": data_dict["system_prompt"]
                })
            for conv in conversations:
                if conv["from"] == "human":
                    messages.append({
                        "role": "user",
                        "content": conv["value"]
                    })
                elif conv["from"] == "gpt":
                    messages.append({
                        "role": "assistant",
                        "content": conv["value"]
                    })
                else:
                    raise ValueError("No such role: {}".format(conv["from"]))
            
            assert messages[-1]["role"] == "assistant"
            messages = messages[:-1]

            output_data.append({
                "messages": messages,
            })

        return output_data

    def read_jsonl_file(self,file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    json_obj = json.loads(line.strip())
                    data.append(json_obj)
                except json.JSONDecodeError:
                    print(f"Error decoding line: {line}")
        return data

    def inference(self):
        save_path = os.path.join(self.save_path, self.save_id)
        os.makedirs(save_path, exist_ok=True)
        
        for param_key, param_dict in self.all_param_dict.items():
            print("[Task] Processing {} ...".format(param_key))
            if param_key not in self.data_map:
                print("there is no data for {}, skip!!!".format(
                    param_key
                ))
                continue
            
            task_save_path = os.path.join(save_path, param_key)
            os.makedirs(task_save_path, exist_ok=True)
            for fidx, fpath in enumerate(self.data_map[param_key]):
                print("[Test {}/{}] inferring file {}".format(
                    fidx, 
                    len(self.data_map[param_key]), 
                    fpath,
                ))
                
                with open(fpath, encoding="utf-8") as fp:
                    all_data = json.load(fp)
                    expanded_data = []
                    for item in all_data:
                        for _ in range(self.samples):
                            expanded_data.append(deepcopy(item))
                    all_data = expanded_data

                file_save_path = os.path.join(
                    task_save_path,
                    "result_{}l".format(
                        fpath.split("/")[-1]
                    )
                )
                start_index = 0
                save_data = []
                if os.path.exists(file_save_path):
                    save_data = self.read_jsonl_file(file_save_path)
                    start_index = len(save_data)
                print("start from {}".format(start_index))

                with open(file_save_path, 'a', encoding='utf-8')as fp_w:
                    batch_data = []
                    for data_idx, data_dict in tqdm(enumerate(all_data[start_index:]), desc="[Process {}]".format(fpath.split("/")[-1])):
                        batch_data.append(data_dict)
                        if len(batch_data) >= self.batch_size or \
                            ((data_idx == (len(all_data[start_index:]) - 1)) and len(batch_data) > 0):
                            
                            output_list = self.model.infer_batch(self._convert_data(batch_data), param_dict)
                            assert len(batch_data) == len(output_list)
                            for save_dict, output_dict in zip(batch_data, output_list):
                                save_dict["model_predictions"] = output_dict
                                json_string_to_save = json.dumps(save_dict, ensure_ascii=False) + '\n'
                                fp_w.write(json_string_to_save)
                            
                            batch_data = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--use_flash_attention", action="store_true")
    parser.add_argument("--acceleration", type=str, default="", choices=["", "vllm"])
    parser.add_argument("--model_class", type=str, default="Qwen2Model")
    parser.add_argument("--task_list", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--save_id", type=str, required=True)
    parser.add_argument("--samples", type=int, required=True)
    
    args = parser.parse_args()

    infer = Infer(
        model_path=args.model_path,
        lora_path=args.lora_path,
        use_flash_attention=args.use_flash_attention,
        acceleration=args.acceleration,
        model_class=args.model_class,
        task_list=args.task_list,
        batch_size=args.batch_size,
        data_path=args.data_path,
        save_path=args.save_path,
        save_id=args.save_id,
        samples=args.samples,
    )
    infer.inference()