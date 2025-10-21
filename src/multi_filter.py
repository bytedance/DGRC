# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os
import json
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from typing import List, Dict, Any
from itertools import combinations
import numpy as np

from utils import (
    read_jsonl_file, write_jsonl_file
)
from models_api import Client_GPT4d1
from PROMPT import MULTI_DIMENSIONAL_BATCH_QA_EVALUATION_PROMPT

def calculate_ifd_for_pair(problem: str, answer: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> float:
    """Calculates the IFD score for a single QA pair."""
    try:
        # 1. Direct Loss (s_direct)
        answer_tokens = tokenizer(answer, return_tensors="pt")
        direct_inputs = {
            "input_ids": answer_tokens.input_ids.to(model.device),
            "attention_mask": answer_tokens.attention_mask.to(model.device),
            "labels": answer_tokens.input_ids.clone().to(model.device)
        }
        with torch.no_grad():
            loss_a = model(**direct_inputs).loss.item()

        # 2. Conditional Loss (s_cond)
        messages = [{"role": "user", "content": problem}, {"role": "assistant", "content": answer}]
        full_inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_tensors="pt")
        
        answer_only_ids = tokenizer(answer, add_special_tokens=False).input_ids
        answer_len = len(answer_only_ids)
        
        labels = full_inputs.clone()
        labels[:, :-answer_len] = -100
        
        conditional_inputs = {"input_ids": full_inputs.to(model.device), "labels": labels.to(model.device)}
        with torch.no_grad():
            cond_loss = model(**conditional_inputs).loss.item()
            
        return cond_loss / loss_a if loss_a > 0 else float("inf")
    except Exception:
        return float("inf") 

def get_embedding_for_pair(text: str, model: AutoModel, tokenizer: AutoTokenizer) -> List[float]:
    """Calculates a sentence embedding for a single text string."""
    try:
        encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            model_output = model(**encoded_input)
            token_embeddings = model_output.last_hidden_state
            input_mask_expanded = encoded_input['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding = (sum_embeddings / sum_mask)[0]
        return embedding.cpu().tolist()
    except Exception:
        return []

class Config:
    """Manages all configuration settings for easy modification."""
    DATA_DIR = "data"
    MODEL_DIR = "model"
    
    RAW_DATA_PATH = os.path.join(DATA_DIR, "MedMCQA_atom_answers_gpt41_qwen7b.json")
    # Intermediate files for robust checkpointing
    FEATURED_DATA_PATH = os.path.join(DATA_DIR, "intermediate_featured_data.jsonl")
    SIMILARITY_FILTERED_PATH = os.path.join(DATA_DIR, "intermediate_sim_filtered_data.jsonl")
    LLM_JUDGE_OUTPUT_PATH = os.path.join(DATA_DIR, "MedMCQA_llm_judge_mid_gpt41_qwen7b.jsonl")
    FINAL_CURRICULUM_PATH = os.path.join(DATA_DIR, "MedMCQA_atom_curriculm_gpt41_qwen7b.jsonl")

    # --- Model Configurations ---
    QWEN_MODEL_ID = os.path.join(MODEL_DIR, "Qwen2.5-7B-Instruct")
    EMBEDDING_MODEL_ID = os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")

    # --- Filtering Parameters ---
    IFD_MIN = 0.35
    IFD_MAX = 1.0
    SIMILARITY_THRESHOLD = 0.85
    LLM_SCORE_THRESHOLD = 13
    
    # --- Processing Parameters ---
    MAX_WORKERS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ATOM_QUESTION_SUFFIX = "\nPlease think carefully step by step, and then give the answer. The output format is as follows: <think> your thinking process </think><answer> your answer </answer>"

# --- Pipeline Functions ---

def load_and_prepare_data(config: Config) -> List[Dict[str, Any]]:
    """Loads raw data and transforms it into a flat list of atomic QA pairs."""
    print("Step 1: Loading and preparing initial data...")
    with open(config.RAW_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    atom_qa_pairs = []

    for item_id_counter, (original_id, content) in enumerate(data.items()):
        for diff_pair in content.get('diff_pair', []):
            atom_questions = diff_pair.get('atom_questions', [])
            atom_answers = diff_pair.get('atom_answers', {})

            for i, q_text in enumerate(atom_questions):
                a_key = list(atom_answers.keys())[i] 
                if a_key in atom_answers:
                    problem_text = q_text + config.ATOM_QUESTION_SUFFIX
                    answer_text = atom_answers[a_key]
                    atom_qa_pairs.append({
                        "id": f"{original_id}-{diff_pair.get('sid', 'N/A')}-{i}",
                        "from": original_id,
                        "problem": problem_text,
                        "answer": answer_text
                    })
    print(f"-> Found {len(atom_qa_pairs)} total atomic QA pairs.")
    return atom_qa_pairs

def calculate_initial_features(raw_pairs: List[Dict[str, Any]], config: Config):
    """Calculates IFD and embeddings, resuming from a checkpoint if available."""
    print("\nStep 2: Calculating initial features (IFD and Embeddings)...")
    
    pairs_to_process = raw_pairs
    if os.path.exists(config.FEATURED_DATA_PATH):
        featured_data = read_jsonl_file(config.FEATURED_DATA_PATH)
        processed_ids = {item['id'] for item in featured_data}
        pairs_to_process = [p for p in raw_pairs if p['id'] not in processed_ids]
        print(f"-> Resuming from checkpoint. {len(processed_ids)} pairs already have features.")
    
    if not pairs_to_process:
        print("-> All features already calculated.")
        return

    print(f"-> Calculating features for {len(pairs_to_process)} new pairs.")
    
    model_ifd = AutoModelForCausalLM.from_pretrained(config.QWEN_MODEL_ID, torch_dtype="auto", device_map="auto").eval()
    tokenizer_ifd = AutoTokenizer.from_pretrained(config.QWEN_MODEL_ID)
    model_emb = AutoModel.from_pretrained(config.EMBEDDING_MODEL_ID).to(config.DEVICE).eval()
    tokenizer_emb = AutoTokenizer.from_pretrained(config.EMBEDDING_MODEL_ID)

    with open(config.FEATURED_DATA_PATH, 'a', encoding='utf-8') as f:
        for pair in tqdm(pairs_to_process, desc="Calculating IFD & Embeddings"):
            pair['ifd'] = calculate_ifd_for_pair(pair['problem'], pair['answer'], model_ifd, tokenizer_ifd)
            embedding_input = pair['problem'] + " " + pair['answer']
            pair['embedding'] = get_embedding_for_pair(embedding_input, model_emb, tokenizer_emb)
            f.write(json.dumps(pair) + '\n')
            f.flush()

def filter_and_deduplicate(featured_data: List[Dict[str, Any]], config: Config) -> List[Dict[str, Any]]:
    """Performs IFD and similarity filtering."""
    print("\nStep 3: Applying IFD and Similarity Filters...")

    # 3a: IFD Filtering
    ifd_filtered = [
        p for p in featured_data 
        if p.get('ifd') and config.IFD_MIN <= p['ifd'] <= config.IFD_MAX
    ]
    print(f"-> After IFD filter: {len(ifd_filtered)} pairs remain (from {len(featured_data)}).")
    
    # 3b: Intra-Problem Similarity Filtering
    grouped_by_from = {}
    for pair in ifd_filtered:
        from_id = pair['from']
        if from_id not in grouped_by_from:
            grouped_by_from[from_id] = []
        grouped_by_from[from_id].append(pair)

    final_filtered_list = []
    for from_id, group in tqdm(grouped_by_from.items(), desc="Similarity Filtering"):
        to_keep = set(range(len(group)))
        
        # Compare all unique pairs within the group
        for i, j in combinations(range(len(group)), 2):
            if i not in to_keep or j not in to_keep:
                continue

            vec_i = np.array(group[i]['embedding'])
            vec_j = np.array(group[j]['embedding'])
            similarity = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))

            if similarity >= config.SIMILARITY_THRESHOLD:
                # Tie-breaker: keep the one with IFD score closer to 1.0
                ifd_i = group[i]['ifd']
                ifd_j = group[j]['ifd']
                if abs(ifd_i - 1.0) <= abs(ifd_j - 1.0):
                    to_keep.discard(j) # Discard j
                else:
                    to_keep.discard(i) # Discard i
        
        final_filtered_list.extend([group[i] for i in sorted(list(to_keep))])

    print(f"-> After Similarity filter: {len(final_filtered_list)} pairs remain.")
    write_jsonl_file(final_filtered_list, config.SIMILARITY_FILTERED_PATH)
    return final_filtered_list

def run_llm_judge(sim_filtered_pairs: List[Dict[str, Any]], config: Config):
    """Scores the filtered QA pairs using an LLM judge in batches."""
    print("\nStep 4: Running LLM judge on the filtered dataset...")
    
    # --- Checkpoint logic to resume processing ---
    pairs_to_judge = sim_filtered_pairs
    if os.path.exists(config.LLM_JUDGE_OUTPUT_PATH):
        already_judged_data = read_jsonl_file(config.LLM_JUDGE_OUTPUT_PATH)
        judged_ids = {item['id'] for item in already_judged_data}
        pairs_to_judge = [p for p in sim_filtered_pairs if p['id'] not in judged_ids]
        print(f"-> Resuming from checkpoint. {len(judged_ids)} pairs already judged. {len(pairs_to_judge)} remaining.")
    else:
        print(f"-> Starting fresh. Judging {len(pairs_to_judge)} pairs.")

    if not pairs_to_judge:
        print("-> All pairs have already been judged.")
        return
        
    client = Client_GPT4d1()

    # --- Group pairs by source problem ID to create batches ---
    grouped_for_judging = {}
    for pair in pairs_to_judge:
        from_id = pair['from']
        if from_id not in grouped_for_judging:
            grouped_for_judging[from_id] = []
        grouped_for_judging[from_id].append(pair)

    print(f"-> Grouped {len(pairs_to_judge)} pairs into {len(grouped_for_judging)} batches.")

    def _create_batch_prompt(batch: List[Dict[str, Any]]) -> str:
        """Formats a list of QA pairs into a single string for the prompt."""
        batch_input_str = ""
        for i, pair in enumerate(batch, 1):
            batch_input_str += f"## Q&A Pair {i}:\n"
            batch_input_str += f"Question: {pair['problem'].split('Please think carefully step by step, and then give the answer.')[0]}\n"
            batch_input_str += f"Answer: {pair['answer']}\n\n"
        return MULTI_DIMENSIONAL_BATCH_QA_EVALUATION_PROMPT.format(
            batch_input=batch_input_str.strip()
        )

    def _process_batch(batch: List[Dict[str, Any]]):
        """Processes a single batch of QA pairs with one API call."""
        if not batch:
            return []
            
        for _ in range(3):
            try:
                prompt = _create_batch_prompt(batch)
                response = client(prompt)
                # The response should be a string representing a JSON array '[{...}, {...}]'
                list_of_scores = json.loads(response)

                # Validate the response from the teacher LLM
                if isinstance(list_of_scores, list) and len(list_of_scores) == len(batch):
                    for i, pair in enumerate(batch):
                        pair['llm_judge_score'] = sum(list_of_scores[i].values())
                    return batch  
            except Exception as e:
                pass
        return None  

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor, \
         open(config.LLM_JUDGE_OUTPUT_PATH, 'a', encoding='utf-8') as f:
        
        # Submit each GROUP as a single task to the thread pool
        future_to_group = {executor.submit(_process_batch, group): group for group in grouped_for_judging.values()}
        
        for future in tqdm(as_completed(future_to_group), total=len(grouped_for_judging), desc="LLM Judging Batches"):
            judged_batch = future.result()
            if judged_batch:
                for pair in judged_batch:
                    f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                f.flush()

def format_final_curriculum(judged_data: List[Dict[str, Any]], config: Config):
    """Applies the final LLM score filter and formats the data for training."""
    print("\nStep 5: Applying final LLM score filter and formatting curriculum...")
    
    # 5a: Final filter based on LLM score
    final_data = [
        p for p in judged_data 
        if p.get('llm_judge_score') and p['llm_judge_score'] >= config.LLM_SCORE_THRESHOLD
    ]
    print(f"-> After LLM score filter: {len(final_data)} pairs remain (from {len(judged_data)}).")

    # 5b: Format for training
    atom_curriculum = []
    for item in final_data:
        conversations = [
            {'from': 'human', 'value': item['problem']},
            {'from': 'gpt', 'value': item['answer']}
        ]
        if isinstance(conversations[1]['value'], str):
            atom_curriculum.append({'id': item['id'], 'conversations': conversations})
    
    write_jsonl_file(atom_curriculum, config.FINAL_CURRICULUM_PATH)
    print(f"-> Final curriculum with {len(atom_curriculum)} items saved to {config.FINAL_CURRICULUM_PATH}")


def main():
    """Main execution pipeline with the new filtering order."""
    config = Config()
    
    # Step 1: Load raw atomic QA pairs.
    raw_pairs = load_and_prepare_data(config)
    
    # Step 2: Calculate IFD and Embeddings (cheap, local computation).
    calculate_initial_features(raw_pairs, config)
    
    # Step 3: Apply IFD and Similarity filters.
    featured_data = read_jsonl_file(config.FEATURED_DATA_PATH)
    sim_filtered_data = filter_and_deduplicate(featured_data, config)
    
    # Step 4: Run LLM Judge on the much smaller, filtered dataset.
    run_llm_judge(sim_filtered_data, config)
    
    # Step 5: Apply the final LLM score filter and save the curriculum.
    judged_data = read_jsonl_file(config.LLM_JUDGE_OUTPUT_PATH)
    format_final_curriculum(judged_data, config)
    
    print("\nAll steps completed successfully!")

if __name__ == "__main__":
    main()
