# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

from tqdm import tqdm

from src.utils import read_jsonl_file, str_to_list
from src.models_api import Client_GPT4d1
from src.PROMPT import ATOM_QUESTION_GENERATION_PROMPT

# --- Configuration ---
DATA_DIR = Path('data')
SOURCE_FILE = DATA_DIR / 'MedMCQA_divergence_gpt41_qwen7b.json'
INTERMEDIATE_FILE = DATA_DIR / 'MedMCQA_atom_questions_mid_gpt41_qwen7b.jsonl'
FINAL_OUTPUT_FILE = DATA_DIR / 'MedMCQA_atom_questions_gpt41_qwen7b.json'
MAX_WORKERS = 12
API_CALL_RETRIES = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_tasks(source_path: Path, intermediate_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads source data, flattens reasoning pairs into a task list,
    and filters out any tasks that have already been processed and saved.

    Returns:
        A tuple containing: (tasks_to_process, original_data_structure).
    """
    logging.info(f"Loading source data from {source_path}...")
    with source_path.open('r', encoding='utf-8') as f:
        source_data = json.load(f)

    all_tasks = []
    for item_id, content in source_data.items():
        for pair in content.get('diff_pair', []):
            task = {
                'id': item_id,
                'problem': content['problem'],
                'label': content['label'],
                'teacher': pair['t'],
                'student': pair['s'],
                'tid': pair['tid'],
                'sid': pair['sid']
            }
            all_tasks.append(task)

    if not intermediate_path.exists():
        logging.info("No intermediate file found. Processing all tasks.")
        return all_tasks, source_data

    logging.info(f"Found intermediate file at {intermediate_path}. Filtering completed tasks.")
    completed_data = read_jsonl_file(str(intermediate_path))
    completed_ids = {(item['id'], item['tid'], item['sid']) for item in completed_data}
    
    tasks_to_process = [
        task for task in all_tasks if (task['id'], task['tid'], task['sid']) not in completed_ids
    ]
    
    logging.info(f"{len(all_tasks)} total tasks found, {len(completed_ids)} already completed.")
    logging.info(f"{len(tasks_to_process)} tasks remaining to be processed.")
    return tasks_to_process, source_data

def process_task(task: Dict[str, Any], client: Client_GPT4d1) -> Optional[Dict[str, Any]]:
    """
    Generates atomic questions for a single task.
    Includes a retry mechanism for robustness.
    """
    prompt = ATOM_QUESTION_GENERATION_PROMPT.format(
        problem=task['problem'],
        cot1=task['teacher'],
        cot2=task['student']
    )
    
    for attempt in range(API_CALL_RETRIES):
        try:
            response_str = client(prompt)
            atomic_questions = str_to_list(response_str)
            
            if isinstance(atomic_questions, list) and atomic_questions:
                task['atom_questions'] = atomic_questions
                return task
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}/{API_CALL_RETRIES} failed for task {task['id']}: {e}")
            
    logging.error(f"Failed to process task {task['id']} (tid={task['tid']}, sid={task['sid']}) after all retries.")
    return None

def merge_results(source_data: Dict[str, Any], intermediate_path: Path, final_path: Path):
    """
    Merges generated atomic questions from the intermediate file back into the
    original data structure and saves the final result to a JSON file.
    """
    logging.info(f"Merging results from {intermediate_path} into the final data structure.")
    processed_items = read_jsonl_file(str(intermediate_path))

    results_lookup = {
        (item['id'], item['tid'], item['sid']): item['atom_questions']
        for item in processed_items if 'atom_questions' in item
    }

    for item_id, content in source_data.items():
        for pair in content.get('diff_pair', []):
            lookup_key = (item_id, pair['tid'], pair['sid'])
            if lookup_key in results_lookup:
                pair['atom_questions'] = results_lookup[lookup_key]

    logging.info(f"Saving final merged data to {final_path}...")
    with final_path.open('w', encoding='utf-8') as f:
        json.dump(source_data, f, ensure_ascii=False, indent=2)

def main():
    """
    Main execution script to generate atomic questions for reasoning pairs.
    """
    tasks_to_run, original_data = prepare_tasks(SOURCE_FILE, INTERMEDIATE_FILE)

    if not tasks_to_run:
        logging.info("No new tasks to process. Proceeding to final merge.")
    else:
        client = Client_GPT4d1()
        logging.info(f"Processing {len(tasks_to_run)} tasks with {MAX_WORKERS} workers...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_task = {executor.submit(process_task, task, client): task for task in tasks_to_run}
            
            with INTERMEDIATE_FILE.open('a', encoding='utf-8') as f:
                progress_bar = tqdm(as_completed(future_to_task), total=len(tasks_to_run), desc="Generating Questions")
                for future in progress_bar:
                    result = future.result()
                    if result:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    merge_results(original_data, INTERMEDIATE_FILE, FINAL_OUTPUT_FILE)
    logging.info("Script finished successfully.")

if __name__ == "__main__":
    main()