# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Callable, Set, Tuple, Optional

from tqdm import tqdm

from src.utils import (
    read_jsonl_file,
    choice_diff_check_4options,
    choice_diff_check_5options,
    choice_diff_check_llm_gpt41,
)

# --- Configuration ---

class Config:
    """Configuration settings for the divergence detection script."""
    STUDENT_DATA_PATH = 'results_student/Qwen2.5-7B-Instruct/train_set/MedMCQA/result_train_samples.jsonl'
    TEACHER_DATA_PATH = 'results_teacher/gpt41/train_set/MedMCQA/result_train_samples.jsonl'
    SAVE_DATA_PATH = 'data/MedMCQA_divergence_gpt41_qwen7b.json'
    MAX_WORKERS = 8
    
def select_judge_function(student_path: str) -> Callable:
    """Selects the appropriate divergence checking function based on the file path."""
    if '7B' in student_path:
        if 'CaseHOLD' in student_path:
            return choice_diff_check_5options
        return choice_diff_check_4options
    return choice_diff_check_llm_gpt41

def load_and_prepare_data(
    student_path: str, teacher_path: str
) -> Dict[str, Any]:
    """
    Loads student and teacher data, finds common problems, and prepares the
    initial data structure for processing.
    """
    print("Step 1: Loading and preparing data...")
    student_data = read_jsonl_file(student_path)
    teacher_data = read_jsonl_file(teacher_path)

    student_ids: Set[str] = {item['id'] for item in student_data}
    teacher_ids: Set[str] = {item['id'] for item in teacher_data}
    common_ids = student_ids & teacher_ids
    print(f"-> Found {len(common_ids)} common problems between teacher and student data.")

    data: Dict[str, Dict[str, Any]] = {
        idx: {'id': idx, 'problem': None, 'label': None, 'teacher': [], 'student': [], 'diff_pair': []}
        for idx in common_ids
    }

    print("-> Populating data structure...")
    for item in teacher_data:
        if item['id'] in common_ids:
            data[item['id']]['problem'] = item['conversations'][0]['value']
            try:
                data[item['id']]['label'] = item['conversations'][1]['value'].split('<answer>')[1].split('</answer>')[0].strip()
            except IndexError:
                data[item['id']]['label'] = None 
            data[item['id']]['teacher'].append(item['model_predictions']['output'])

    for item in student_data:
        if item['id'] in common_ids:
            data[item['id']]['student'].append(item['model_predictions']['output'])
            
    return data

def process_divergence_pairs(
    data: Dict[str, Any], judge_function: Callable, max_workers: int
) -> Dict[str, Any]:
    """
    Concurrently processes all teacher-student pairs to find divergences.
    """
    print("\nStep 2: Detecting divergences concurrently...")
    
    def _judge_worker(
        problem: str, teacher_cot: str, student_cot: str, tid: int, sid: int
    ) -> Optional[Dict[str, Any]]:
        try:
            if judge_function(problem, teacher_cot, student_cot):
                return {'tid': tid, 'sid': sid, 't': teacher_cot, 's': student_cot}
        except Exception:
            pass
        return None

    tasks_to_submit = []
    for idx, content in data.items():
        problem = content['problem']
        for tid, teacher_cot in enumerate(content['teacher']):
            for sid, student_cot in enumerate(content['student']):
                tasks_to_submit.append(
                    (idx, problem, teacher_cot, student_cot, tid, sid)
                )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_judge_worker, task[1], task[2], task[3], task[4], task[5]): task[0]
            for task in tasks_to_submit
        }

        for future in tqdm(as_completed(future_to_idx), total=len(tasks_to_submit), desc="Judging Pairs"):
            idx = future_to_idx[future]
            result_pair = future.result()
            if result_pair:
                data[idx]['diff_pair'].append(result_pair)
                
    return data

def main():
    """Main execution pipeline for divergence detection."""
    config = Config()

    # Step 1: Load and structure the data
    data = load_and_prepare_data(config.STUDENT_DATA_PATH, config.TEACHER_DATA_PATH)

    # Step 2: Select the appropriate judge function
    judge_function = select_judge_function(config.STUDENT_DATA_PATH)

    # Step 3: Run the divergence detection process in parallel
    processed_data = process_divergence_pairs(data, judge_function, config.MAX_WORKERS)

    # Step 4: Save the final results
    print(f"\nStep 3: Saving final data to {config.SAVE_DATA_PATH}...")
    with open(config.SAVE_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
        
    print("Divergence detection complete.")

if __name__ == "__main__":
    main()
