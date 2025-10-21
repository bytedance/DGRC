# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Set, Tuple

from tqdm import tqdm

from src.utils import read_jsonl_file, write_jsonl_file, extract_dict_from_string
from src.models_api import Client_GPT4d1
from src.PROMPT import ATOM_ANSWER_GENERATION_PROMPT

# --- Configuration ---
class Config:
    """Configuration settings for the script."""
    DATA_PATH = 'data/MedMCQA_atom_questions_gpt41_qwen7b.json'
    MID_OUTPUT_FILE = 'data/MedMCQA_atom_answers_mid_gpt41_qwen7b.jsonl'
    FINAL_OUTPUT_FILE = 'data/MedMCQA_atom_answers_gpt41_qwen7b.json'
    MAX_WORKERS = 24
    MAX_RETRIES = 3

def load_and_flatten_tasks(data_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Loads the source JSON data and flattens the nested structure into a list of
    processing tasks.

    Args:
        data_path: Path to the source JSON file.

    Returns:
        A tuple containing the original source data and the list of tasks.
    """
    print("Step 1: Loading and preparing tasks...")
    with open(data_path, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    tasks = []
    for item_id, content in source_data.items():
        for diff_pair in content.get('diff_pair', []):
            task = {
                'id': item_id,
                'problem': content.get('problem'),
                'label': content.get('label'),
                'teacher_cot': diff_pair.get('t'),
                'student_cot': diff_pair.get('s'),
                'tid': diff_pair.get('tid'),
                'sid': diff_pair.get('sid'),
                'atom_questions': diff_pair.get('atom_questions', [])
            }
            tasks.append(task)
            
    print(f"-> Created {len(tasks)} processing tasks.")
    return source_data, tasks

def filter_completed_tasks(
    all_tasks: List[Dict[str, Any]],
    checkpoint_file: str
) -> List[Dict[str, Any]]:
    """
    Filters out tasks that have already been processed by checking a checkpoint file.
    """
    if not os.path.exists(checkpoint_file):
        print("-> No checkpoint file found. Processing all tasks.")
        return all_tasks

    print(f"-> Checkpoint file found at {checkpoint_file}. Filtering completed tasks...")
    processed_data = read_jsonl_file(checkpoint_file)
    processed_keys: Set[Tuple[str, int, int]] = {
        (item['id'], item['tid'], item['sid']) for item in processed_data
    }
    
    tasks_to_run = [
        task for task in all_tasks
        if (task['id'], task['tid'], task['sid']) not in processed_keys
    ]
    
    print(f"-> {len(processed_keys)} tasks already completed. {len(tasks_to_run)} tasks remaining.")
    return tasks_to_run

def process_task(task: Dict[str, Any], client: Client_GPT4d1) -> Optional[Dict[str, Any]]:
    """
    Processes a single task by formatting a prompt, calling the LLM, and parsing the response.
    """
    if not task.get('atom_questions'):
        return None

    formatted_questions = "\n".join([
        f"Question{i+1}: {q}" for i, q in enumerate(task['atom_questions'])
    ])
    
    prompt = ATOM_ANSWER_GENERATION_PROMPT.format(atom_questions=formatted_questions)

    for _ in range(Config.MAX_RETRIES):
        try:
            response = client(prompt)
            atom_answers = extract_dict_from_string(response)
            if atom_answers:
                task['atom_answers'] = atom_answers
                return task
        except Exception:
            continue
            
    return None 

def merge_results_into_source(
    source_data: Dict[str, Any],
    mid_output_file: str
) -> Dict[str, Any]:
    """
    Merges the generated atomic answers from the intermediate file back into the
    original nested data structure.
    """
    print("\nStep 4: Merging results back into source data...")
    mid_data = read_jsonl_file(mid_output_file)
    
    diff_pair_map = {}
    for item_id, content in source_data.items():
        for diff_pair in content.get('diff_pair', []):
            key = (item_id, diff_pair['tid'], diff_pair['sid'])
            diff_pair_map[key] = diff_pair 

    for item in tqdm(mid_data, desc="Merging"):
        if 'atom_answers' in item:
            key = (item['id'], item['tid'], item['sid'])
            if key in diff_pair_map:
                diff_pair_map[key]['atom_answers'] = item['atom_answers']

    return source_data

def main():
    """Main execution pipeline."""
    config = Config()
    client = Client_GPT4d1()

    # Step 1: Load and prepare all potential tasks
    source_data, all_tasks = load_and_flatten_tasks(config.DATA_PATH)
    
    # Step 2: Filter out tasks that have already been completed
    tasks_to_run = filter_completed_tasks(all_tasks, config.MID_OUTPUT_FILE)

    # Step 3: Concurrently process the remaining tasks
    if tasks_to_run:
        print(f"\nStep 3: Processing {len(tasks_to_run)} tasks...")
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor, \
             open(config.MID_OUTPUT_FILE, 'a', encoding='utf-8') as f:
            
            future_to_task = {executor.submit(process_task, task, client): task for task in tasks_to_run}
            
            for future in tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="Generating Answers"):
                result = future.result()
                if result:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:
        print("\nStep 3: No new tasks to process.")

    # Step 4: Merge the results from the intermediate file into the source data
    final_data = merge_results_into_source(source_data, config.MID_OUTPUT_FILE)
    
    # Step 5: Save the final, complete data structure
    print(f"Step 5: Saving final data to {config.FINAL_OUTPUT_FILE}...")
    with open(config.FINAL_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
