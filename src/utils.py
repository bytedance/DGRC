# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import ast
from typing import List, Any
from PROMPT import ANSWER_CONFLICT_DETECTION_PROMPT
from models_api import Client_GPT4d1

client = Client_GPT4d1()

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line.strip())
                data.append(json_obj)
            except json.JSONDecodeError:
                print(f"Error decoding line: {line}")
    return data

def write_jsonl_file(data, path):
    import json
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def choice_diff_check_4options(problem, cot1, cot2):
    try:
        ans1 = cot1.split('<answer>')[1].split('</answer>')[0].strip()
    except:
        ans1 = None
    try:
        ans2 = cot2.split('<answer>')[1].split('</answer>')[0].strip()
    except:
        ans2 = None
    if ans1 and ans2 and ans1.lower() in ['a','b','c','d'] and ans2.lower() in ['a','b','c','d'] and ans1.lower() != ans2.lower():
        return True
    else:
        return False

def choice_diff_check_5options(problem, cot1, cot2):
    try:
        ans1 = cot1.split('<answer>')[1].split('</answer>')[0].strip()
    except:
        ans1 = None
    try:
        ans2 = cot2.split('<answer>')[1].split('</answer>')[0].strip()
    except:
        ans2 = None
    if ans1 and ans2 and ans1.lower() in ['a','b','c','d','e'] and ans2.lower() in ['a','b','c','d','e'] and ans1.lower() != ans2.lower():
        return True
    else:
        return False
    
def choice_diff_check_llm_gpt41(problem, cot1, cot2):
    prompt = ANSWER_CONFLICT_DETECTION_PROMPT.format(problem=problem, cot1=cot1, cot2=cot2)
    for i in range(3):
        try:
            response = client(prompt)
            if response.strip().upper() == '<INCONSISTENT>':
                return True
            elif response.strip().upper() == '<CONSISTENT>':
                return False
        except Exception as e:
            continue
    return True

def str_to_list(s: str) -> List[Any]:
    try:
        result = ast.literal_eval(s)
        if not isinstance(result, list):
            return None
        return result
    except (SyntaxError, ValueError) as e:
        return None

def extract_dict_from_string(s):
    start_index = s.find('{')
    if start_index == -1:
        return None

    brace_count = 0
    end_index = -1
    for i in range(start_index, len(s)):
        if s[i] == '{':
            brace_count += 1
        elif s[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_index = i
                break

    if end_index == -1:
        return None

    dict_str = s[start_index:end_index + 1]
    try:
        result = json.loads(dict_str)
        return result
    except json.JSONDecodeError:
        try:
            result = ast.literal_eval(dict_str)
            return result
        except (SyntaxError, ValueError):
            return None
