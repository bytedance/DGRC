# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any

from tqdm import tqdm

from utils import read_jsonl_file, write_jsonl_file
from models_api import Client_GPT4d1
from PROMPT import VERIFIED_COT_CURATION_PROMPT

# --- Configuration ---
DATA_DIR = Path('data')
SOURCE_DATA_PATH = DATA_DIR / 'MedMCQA_atom_answers_gpt41_qwen7b.json'
FILTERED_PAIRS_PATH = DATA_DIR / 'MedMCQA_llm_judge_mid_gpt41_qwen7b.jsonl'
VERIFIED_OUTPUT_PATH = DATA_DIR / 'MedMCQA_verified_cots_curriculum_gpt41_qwen7b.jsonl'
NO_DIVERGENCE_OUTPUT_PATH = DATA_DIR / 'MedMCQA_no_divergence_cots_curriculum_gpt41_qwen7b.jsonl'

# Model and processing parameters.
MAX_WORKERS = 8
API_CALL_RETRIES = 3
MIN_JUDGE_SCORE = 13 # Minimum score for an atomic pair to be considered valid.

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_verification_tasks(source_data: Dict[str, Any], atomic_pairs_path: Path) -> Dict[str, Dict]:
    """
    Filters source data for divergent pairs and enriches them with high-quality atomic knowledge.

    Returns:
        A dictionary of tasks, where each key is an item ID and the value contains
        the problem, original answer, and a list of atomic Q&A pairs.
    """
    logging.info("Preparing tasks for verification from divergent pairs...")
    
    # 1. Initialize tasks with problems that have divergent reasoning pairs.
    tasks = {
        item_id: {
            'problem': content['problem'],
            'answer': content['teacher'][0], 
            'atom_knowledge': []
        }
        for item_id, content in source_data.items() if content.get('diff_pair')
    }

    # 2. Load and process the filtered atomic Q&A pairs.
    if not atomic_pairs_path.exists():
        logging.warning(f"Atomic pairs file not found at {atomic_pairs_path}. No knowledge will be added.")
        return tasks
        
    atomic_qa_pairs = read_jsonl_file(str(atomic_pairs_path))

    # 3. Merge high-quality atomic knowledge into the corresponding tasks.
    for pair in atomic_qa_pairs:
        item_id = pair.get('from')
        if item_id in tasks and pair.get('llm_judge_score', 0) >= MIN_JUDGE_SCORE:
            try:
                # Extract and clean the relevant text fields.
                clean_question = pair['problem'].split('\nPlease think carefully step by step')[0]
                clean_answer = pair['answer'].split('<answer>')[1].split('</answer>')[0].strip()
                
                tasks[item_id]['atom_knowledge'].append({
                    'atom_question': clean_question,
                    'atom_answer': clean_answer
                })
            except (KeyError, IndexError):
                logging.warning(f"Skipping malformed atomic pair for item ID {item_id}.")

    logging.info(f"Prepared {len(tasks)} problems for CoT verification.")
    return tasks

def _format_atomic_facts(knowledge_list: List[Dict[str, str]]) -> str:
    """Helper function to format a list of Q&A pairs into a single paragraph of facts."""
    if not knowledge_list:
        return "No atomic facts provided."
    
    facts = []
    for item in knowledge_list:
        fact_statement = item['atom_answer']
        if not fact_statement.endswith('.'):
            fact_statement += '.'
        facts.append(fact_statement)

    return " ".join(facts)

def _verify_single_item(item_id: str, task_data: Dict, client: Client_GPT4d1) -> tuple[str, bool]:
    """
    Worker function to verify a single CoT against its atomic facts using the teacher LLM.
    """
    # If no atomic knowledge was found/merged, the CoT cannot be contradicted.
    if not task_data.get('atom_knowledge'):
        return item_id, True

    prompt = VERIFIED_COT_CURATION_PROMPT.format(
        original_question=task_data['problem'],
        reasoning_chain=task_data['answer'],
        atomic_facts=_format_atomic_facts(task_data['atom_knowledge'])
    )

    for attempt in range(API_CALL_RETRIES):
        try:
            response = client(prompt)
            is_consistent = "<CONSISTENT>" in response.upper()
            return item_id, is_consistent
        except Exception as e:
            logging.warning(f"API call failed for item {item_id} (attempt {attempt + 1}): {e}")
    
    return item_id, False

def execute_verification(tasks: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Manages the parallel execution of the CoT verification process.
    """
    if not tasks:
        logging.info("No tasks to verify.")
        return {}
        
    logging.info(f"Starting verification for {len(tasks)} CoTs using {MAX_WORKERS} workers...")
    client = Client_GPT4d1()
    verified_cots = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(_verify_single_item, item_id, data, client): item_id for item_id, data in tasks.items()}
        
        progress_bar = tqdm(as_completed(future_to_id), total=len(tasks), desc="Verifying CoTs")
        for future in progress_bar:
            item_id, is_consistent = future.result()
            if is_consistent:
                verified_cots[item_id] = tasks[item_id]

    return verified_cots

def save_formatted_data(data_dict: Dict, output_path: Path):
    """Formats and saves the final data to a JSONL file."""
    data_to_save = [
        {'id': item_id, 'conversations': [
            {'from': 'human', 'value': content['problem']},
            {'from': 'gpt', 'value': content['answer']}
        ]}
        for item_id, content in data_dict.items()
    ]
    write_jsonl_file(data_to_save, str(output_path))
    logging.info(f"Successfully saved {len(data_to_save)} items to {output_path}")

def main():
    """Main script execution flow."""
    logging.info("Starting CoT verification and curriculum curation pipeline.")
    
    # 1. Load the source data.
    with SOURCE_DATA_PATH.open('r', encoding='utf-8') as f:
        source_data = json.load(f)

    # 2. Handle non-divergent data: format and save them directly.
    logging.info("Processing non-divergent items...")
    non_divergent_to_save = [
        {
            'id': item_id,
            'conversations': [
                {'from': 'human', 'value': content['problem']},
                {'from': 'gpt', 'value': content['teacher'][0]}  
            ]
        }
        for item_id, content in source_data.items() if not content.get('diff_pair')
    ]

    if non_divergent_to_save:
        write_jsonl_file(non_divergent_to_save, str(NO_DIVERGENCE_OUTPUT_PATH))
        logging.info(f"Successfully saved {len(non_divergent_to_save)} non-divergent items.")
    else:
        logging.info("No non-divergent items found to process.")

    # 3. Prepare tasks for the CoTs that require verification.
    tasks_to_verify = prepare_verification_tasks(source_data, FILTERED_PAIRS_PATH)
    
    # 4. Run the parallel verification process.
    verified_data = execute_verification(tasks_to_verify)
    
    # 5. Save the successfully verified CoTs using the helper function.
    if verified_data:
        logging.info(f"Found {len(verified_data)} verified CoTs out of {len(tasks_to_verify)} candidates.")
        save_formatted_data(verified_data, VERIFIED_OUTPUT_PATH)
    else:
        logging.info("No CoTs were verified in this run.")
        
    logging.info("Pipeline finished successfully.")

if __name__ == "__main__":
    main()