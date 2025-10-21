# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

ANSWER_CONFLICT_DETECTION_PROMPT= """
# Role and Task
You are a precise answer verifier. Your task is to determine if Answer A and Answer B are semantically consistent and provide the same final conclusion for the given Question.

# Rules
- **Context is Key:** Your judgment must be based entirely on the context provided by the Question.
- **Semantic Equivalence:** Consider two answers CONSISTENT if they mean the same thing, even if the wording is different. This includes synonyms, paraphrasing, and different but equivalent numerical representations (e.g., "50%", "0.5").

# Strict Output Format
Your output MUST be one of the following two words, and nothing else. Do not provide any explanation or additional text.
- If consistent, output ONLY the string: <CONSISTENT>
- If inconsistent, output ONLY the string: <INCONSISTENT>

---
Question:
{problem}

Answer A:
{cot1}

Answer B:
{cot2}

Your Output:
"""


ATOM_QUESTION_GENERATION_PROMPT = """
# Role
Act as an expert in logical analysis. Your specialty is deconstructing and comparing complex reasoning processes to identify the fundamental points of divergence that lead to different conclusions.

# Task
I will provide you with an Original Question and two distinct Reasoning Processes (A and B) from two different models, which have resulted in conflicting answers. Your task is to perform a deep analysis of both reasoning chains, pinpoint the exact discrepancies between them, and formulate these points of divergence into a set of Atomic Questions.

# Core Requirements for Atomic Questions
Each atomic question you generate must strictly adhere to the following rules:

1. Atomicity and Independence: Each question must be the smallest possible logical unit and be completely independent of all other questions. A person should be able to understand and answer any single question on its own without needing to read any others.

2. Focus on Discrepancy: Each question must target a specific, concrete point of disagreement between the two reasoning processes. Examples include differences in factual claims, conflicting logical steps, or contradictory underlying assumptions.

3. Self-Contained: If background context is necessary to understand the question, concisely embed that context within the question itself.

4. Verifiability: Each question must be phrased in a way that it can be answered definitively through factual verification, a clear logical judgment, or a straightforward calculation.

# Input Format

## Original Question
{problem}

## Reasoning Process A
{cot1}

## Reasoning Process B
{cot2}

Please output in the following list format:
["Atomic Question 1","Atomic Question 2","Atomic Question 3",...]

** NOTE **
1. Don't solve the original question, just output the atomic questions.
2. When outputting atomic questions, strictly follow the specified format and do not output any extra symbols or thought processes.
3. Ensure the atomicity of atomic questions. That is, do not use pronouns such as Process A/B or Option A/B/C/D. Just look at the atomic question itself to answer it.
"""

ATOM_ANSWER_GENERATION_PROMPT = """
# Role
You are a logical reasoning engine. Your task is to answer a list of atomic questions provided below. You must process each question sequentially and meticulously.

# Task
For each question, you must generate a response that includes your detailed thinking process and the final answer. This response for each question must be formatted as a single string with the following structure: <think>Your detailed, step-by-step thinking process goes here.</think><answer>Your final, conclusive answer goes here.</answer>.

Your final output for the entire list of questions must be a single JSON object. Do not write any text or explanations outside of this JSON object.

# Key Requirements:

## Sequential Order: You must answer the questions in the exact order they are presented.

## Strict Formatting: The final output must be a JSON object. The keys of the object must be "Answer_to_Question_1", "Answer_to_Question_2", "Answer_to_Question_3", and so on. The value for each key must be the string containing the <think> and <answer> tags.

## Detailed and Clear: Both the thinking process and the final answer must be detailed, accurate, and clear.

# Example
If the input list of questions is:
Question1: What is the primary cause of Earth's seasons?
Question2: Calculate the sum of 15 and 27.

Your output must be exactly this:
{{
  "Answer_to_Question1": "<think>The user is asking about the cause of Earth's seasons. The common misconception is that it's due to the Earth's changing distance from the Sun. However, the primary cause is the tilt of the Earth's rotational axis, which is about 23.5 degrees. When a hemisphere is tilted towards the Sun, it receives more direct sunlight and experiences summer. When it's tilted away, it receives less direct sunlight and experiences winter. I will state this axial tilt as the main reason.</think><answer>The tilt of the Earth's rotational axis.</answer>",
  "Answer_to_Question2": "<think>The user wants me to perform a simple addition. The two numbers are 15 and 27. I will add the units place: 5 + 7 = 12. I carry over the 1. Then I will add the tens place, including the carry-over: 1 + 1 + 2 = 4. The result is 42. I will provide this as the final answer.</think><answer>42</answer>"
}}

Now, please process the following list of atomic questions:
{atom_questions}
"""

MULTI_DIMENSIONAL_BATCH_QA_EVALUATION_PROMPT = """
# Role
You are an expert evaluator. Your task is to provide rigorous, objective quality scores for a batch of given atomic question-answer (Q&A) pairs.

# Scoring Dimensions and Rubric
For each Q&A pair, you will provide a score from 0 to 2 for each of the following dimensions:
1.  **Clarity (0-2):** How clear and unambiguous is the Q&A? (2: Perfectly clear, 1: Mostly clear, 0: Vague)
2.  **Completeness (0-2):** Does the answer fully address the question? (2: Comprehensive, 1: Addresses main point, 0: Incomplete)
3.  **Structure (0-2):** Is the answer well-structured? (2: Well-organized, 1: Acceptable, 0: Unstructured)
4.  **Credibility (0-2):** Is the answer factually correct and free of hallucinations? (2: Fully credible, 1: Minor inaccuracies, 0: Factually incorrect)
5.  **Knowledge Richness (0-2):** Does the answer provide sufficient, self-contained knowledge? (2: Richly informative, 1: Lacks context, 0: Too simplistic)
6.  **Logicality (0-2):** Is the answer logically sound? (2: Perfectly sound, 1: Minor flaws, 0: Illogical)
7.  **Instruction Following (0-2):** Does the answer adhere to all instructions? (2: Perfectly follows, 1: Minor deviations, 0: Fails to follow)

# Batch Input
You will receive a numbered list of Q&A pairs to evaluate.
{batch_input}

# Strict Output Format
You must provide your evaluation as a JSON array, where each element is a dictionary of scores. The order of the JSON objects in the array MUST correspond to the order of the Q&A pairs in the input. Your output must be ONLY the JSON array and nothing else.
"""

VERIFIED_COT_CURATION_PROMPT = """
# Role
You are a meticulous Factual Consistency Verifier. Your task is to act as an impartial judge, determining if a given Reasoning Chain contains any statements that contradict a provided set of established Atomic Facts.

# Task
I will provide you with an Original Question, a Reasoning Chain to be evaluated, and a list of Atomic Facts (in Q&A format) that are to be considered the absolute ground truth. Your sole task is to check for contradictions.

# Core Rules for Verification
- Atomic Facts are Ground Truth: The provided Atomic Q&A pairs are the definitive source of truth for this task. The Reasoning Chain must be evaluated strictly against them.
- Identify Contradictions: A contradiction exists if any part of the Reasoning Chain makes a claim that is logically or factually inconsistent with any of the Atomic Facts. This includes direct factual errors, flawed logical steps, or incorrect calculations.
- Implicit vs. Explicit: The contradiction can be explicit (e.g., Reasoning says "total is 30%", Fact says "total is less than 30%") or implicit (e.g., a calculation in the Reasoning violates a principle explained in a Fact).

# Input Format
## Original Question:
{original_question}

## Reasoning Chain to Verify:
{reasoning_chain}

## Atomic Facts: 
{atomic_facts}

# Strict Output Format
Your output MUST be one of the following two words, and nothing else. Do not provide any explanation.
- If the Reasoning Chain is fully consistent with ALL Atomic Facts, output ONLY "<CONSISTENT>"
- If the Reasoning Chain contradicts ANY of the Atomic Facts, output ONLY "<INCONSISTENT>"

Your Output:
"""