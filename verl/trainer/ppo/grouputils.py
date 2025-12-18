# Copyright 2025 TTRL Team (https://arxiv.org/abs/2504.16084)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List
from collections import Counter, defaultdict
# import torch
import random
import numpy as np
from verl.utils.reward_score.ttrl_math import extract_answer, simplify_expression_string, grade

def select_top_k_per_prompt(data, n_votes_per_prompt, n_samples_per_prompt):
    """
    Select the first k rollouts per prompt, used for TTRL downsampling.
    """
    assert len(data) % n_votes_per_prompt == 0, "data length must be divisible by n_votes_per_prompt"
    num_prompts = len(data) // n_votes_per_prompt

    selected_indices = []
    for i in range(num_prompts):
        start = i * n_votes_per_prompt
        selected_indices.extend(range(start, start + n_samples_per_prompt))

    return data[selected_indices]


# === TTRL Ground Truth Manipulation ===


def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch.
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch

def apply_group_conf_gt(batch, group_conf_gt, n):
    
    # print(f"len(batch): {len(batch)}, len(group_conf_gt): {len(group_conf_gt)}")
    assert len(batch) == len(group_conf_gt), "batch length must be equal to the group_conf_gt length"

    prompt_num = len(batch) // n

    prompt_gt_diversity = []

    for i in range(prompt_num):
        start = i * n
        prompt_gt = []
        for j in range(n):
            prompt_gt.append(group_conf_gt[start + j])
        prompt_gt_diversity.append(len(set(prompt_gt)))

    prompt_gt_diversity = [i for i in prompt_gt_diversity for _ in range(n)]

    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = group_conf_gt[i]
        data_item.non_tensor_batch["reward_model"]["majority_gt"] = group_conf_gt[i]
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    batch.non_tensor_batch["prompt_gt_diversity"] = np.array(prompt_gt_diversity, dtype=int)
    assert len(batch.non_tensor_batch["prompt_gt_diversity"]) == len(batch), "prompt_gt_diversity length must be equal to the batch length"

    return batch

def compute_conf_gt(gen_batch_output, n, group_vote_num, group_size, tokenizer):
    """
    Apply the majority vote ground truth to the batch.
    """
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    num_prompts = len(gen_batch_output) // n
    # assert len(batch) == num_prompts, "batch length must be equal to the number of prompts"
    assert group_vote_num <= n, "group_vote_num must be less than or equal to n"
    assert n % group_size == 0, "n must be divisible by group_size"

    confidence = gen_batch_output.non_tensor_batch["confs"] # ndarray(shape=(len(gen_batch_output)))

    model_outputs = []

    for i in range(num_prompts):
        start = i * n
        for j in range(n):
            data_item = gen_batch_output[start + j]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            model_outputs.append(response_str)

    assert len(confidence) == len(model_outputs), "length of condfidence and model_outputs must be equal"

    majority_gt_list = _batch_conf_vote(model_outputs, n, group_vote_num, group_size, confidence)
    
    return majority_gt_list

def _batch_conf_vote(model_outputs: List[str], n: int, group_vote_num: int, group_size: int, confidence: np.ndarray) -> tuple[List[str]]:
    majority_gt_list = []
    n_prompts = len(model_outputs) // n
    group_num = n // group_size
    resp_conf = [(i,j) for i, j in zip(model_outputs, confidence)]

    for i in range(n_prompts):
        prompt_resp_conf = resp_conf[i * n:(i + 1) * n] # list: (response, confidence) of a prompt
        # each group of a prompt will be voted separately
        for j in range(group_num):
            # random sample group_vote_num elements from prompt_resp_conf
            sampled_resp_conf = random.sample(prompt_resp_conf, group_vote_num)
            prompt_group_outputs = [r for r, _ in sampled_resp_conf]
            prompt_group_confidence = [c for _, c in sampled_resp_conf]
            prompt_group_gt = _majority_vote_with_confidence(prompt_group_outputs, prompt_group_confidence)
            majority_gt_list.append(prompt_group_gt)

    majority_gt_list = [x for x in majority_gt_list for _ in range(group_size)]

    assert len(majority_gt_list) == len(model_outputs), "majority_gt_list length must be equal to the number of model outputs"
    
    return majority_gt_list

def _majority_vote_with_confidence(model_outputs: List[str], confidence: np.ndarray) -> str:
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None"

    score = defaultdict(float)

    for item, conf in zip(model_answers, confidence):
        score[item] += conf
    
    most_confidence_answer = max(score, key=score.get)

    return most_confidence_answer

# === Metrics Computation ===


def compute_ttrl_metrics(batch, n, group_size, n_votes_per_subgroup):
    """
    Compute the TTRL metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    group_conf_reward = []
    gt_reward = []
    group_conf_label = []
    gt_label = []

    for i in range(len(batch)):
        data_item = batch[idx[i]]
        group_conf_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        group_conf_label.append(data_item.non_tensor_batch["reward_model"]["majority_gt"]) # 要改
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"]) 

    # Fix argument ordering: avoid positional args after keyword
    ttrl_metrics = _batch_compute_ttrl_metrics(
        group_conf_reward,
        gt_reward,
        group_conf_label,
        gt_label,
        n,
        group_size,
        n_votes_per_subgroup,
    )
    prompt_gt_diversity = batch.non_tensor_batch["prompt_gt_diversity"]
    prompt_gt_diversity = sum(prompt_gt_diversity) / len(prompt_gt_diversity)
    ttrl_metrics["vote_diversity"] = prompt_gt_diversity

    return ttrl_metrics

# TODO: TO BE TESTED
def _batch_compute_ttrl_metrics(
    group_conf_reward: List[float],
    gt_reward: List[float],
    group_conf_label: List[str],
    gt_label: List[str],
    n: int,
    group_size: int,
    n_votes_per_subgroup: int,
):
    """
    Compute the TTRL metrics for batch inputs.
    """
    assert len(group_conf_reward) == len(gt_reward) == len(group_conf_label) == len(gt_label)
    assert len(group_conf_reward) % n == 0
    n_prompts = len(group_conf_reward) // n
    ttrl_metrics = []
    group_num = n // group_size
    
    for i in range(n_prompts):
        prompt_group_conf_reward = group_conf_reward[i * n:(i + 1) * n]
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_group_conf_label = group_conf_label[i * n:(i + 1) * n]
        prompt_gt_label = gt_label[i * n:(i + 1) * n]

        # c1 = Counter(prompt_group_conf_label).most_common(1)[0][1]
        # c2 = Counter(prompt_gt_label).most_common(1)[0][1]
        # c3 = n
        
        for j in range(group_num):
            single_group_conf_label = prompt_group_conf_label[j * group_size:(j + 1) * group_size]
            single_group_gt_label = prompt_gt_label[j * group_size:(j + 1) * group_size]
            c1 = Counter(single_group_conf_label).most_common(1)[0][1]
            c2 = Counter(single_group_gt_label).most_common(1)[0][1]
            c3 = group_size

            assert Counter(single_group_conf_label).most_common(1)[0][1] == group_size
            assert Counter(single_group_gt_label).most_common(1)[0][1] == group_size

        ttrl_metric = _prompt_compute_ttrl_metrics(prompt_group_conf_reward, prompt_gt_reward, prompt_group_conf_label, prompt_gt_label, group_size)
        ttrl_metrics.append(ttrl_metric)

    # Compute the average metrics
    ttrl_metrics = {k: sum(d[k] for d in ttrl_metrics) / len(ttrl_metrics) for k in ttrl_metrics[0]}

    return ttrl_metrics

def _prompt_compute_ttrl_metrics(
    prompt_group_conf_reward: List[float],
    prompt_gt_reward: List[float],
    prompt_group_conf_labels: List[str],
    gt_label: List[str],
    group_size: int,
    ):    
    assert len(prompt_group_conf_reward) == len(prompt_gt_reward)
    # take first element in each group

    _ = []
    for i, j in enumerate(prompt_group_conf_labels):
        if i % group_size == 0:
            _.append(j)
    prompt_group_conf_labels = _

    hit_rate = sum([1 for i, j in zip(prompt_group_conf_labels, gt_label) if grade(i, j)]) / len(prompt_group_conf_labels)
    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(prompt_group_conf_reward, prompt_gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(prompt_group_conf_reward)
    
    ttrl_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "group_confs_voting_reward": sum(prompt_group_conf_reward) / len(prompt_group_conf_reward),
        "ground_truth_reward": sum(prompt_gt_reward) / len(prompt_gt_reward),
        f"pass@{len(prompt_group_conf_reward)}": 1.0 if sum(prompt_gt_reward) >= 1 else 0.0,
    }
    return ttrl_metric
