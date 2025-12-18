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
import numpy as np
from transformers import AutoTokenizer
from typing import List
from collections import Counter, defaultdict
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


# === Step Split and Confidence Calculation ===

def split_steps(ids: list, split_token: str, tokenizer: AutoTokenizer) -> list[tuple[int, int]]:
    """
    Split the tokens into steps and return the start and end indices of each step.
    Args:
        ids: list of token_ids
        split_token: the token to split the steps
    Returns:
        list of tuples of (start, end) indices of each step
    """
    # print(f"the split token is {repr(split_token)}")
    # print(f"first five tokens ids are {ids[:5]}")
    assert type(split_token) != None, "the split token should not be None"
    assert type(ids) == list, "the ids should be a list"
    

    tokens = tokenizer.convert_ids_to_tokens(ids)
    start = 0
    result = []
    for i, token in enumerate(tokens):
        # NOTE: When using V1, the generate method may output a token ID that exceeds the vocabulary size, which then cannot be converted and results in None.
        if type(token) != str:
            continue
            print(f"the token is {repr(token)}")
            
        # assert token != None, "the token should not be None"
        if split_token in token:
            result.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        result.append((start, len(tokens)))
    return result

def compute_confidence(logprobs):
    """
    Compute confidence score from logprobs
    Args:
        logprobs: list of logprobs
    Returns:
        list of confidence scores
    """
    confs = []
    for token_logprobs in logprobs:
        if token_logprobs:
            # vLLM returns a dict of {token_id: Logprob object}
            # Get the selected token's logprob (the one with highest probability)
            mean_logprob = np.mean([lp.logprob for lp in token_logprobs.values()])
            confs.append(round(-mean_logprob, 3))
    return confs

def calculate_avg_step_conf(steps, logprobs) -> float:
    """
    Calculate the average step confidence from the logprobs
    Args:
        steps: list of tuples of (start, end) indices of each step
        logprobs: list of logprobs
    Returns:
        float: the average step confidence
    """
    avg_step_conf = 0
    for start, end in steps:
        confidences = compute_confidence(logprobs[start:end])
        step_conf = np.mean(confidences)
        avg_step_conf += step_conf
    avg_step_conf /= len(steps)
    return avg_step_conf

# === Group Conf Ground Truth Manipulation ===

def distribute_group_conf_gt(batch, n, group_size):

    for i in range(len(batch)):
        group_idx = i % n // group_size
        data_item = batch[i]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = data_item.non_tensor_batch["reward_model"]["group_conf_gt"][group_idx]
    
    return batch

# NOTE: A copy from ttrl_utils.py/apply_original_gt
def apply_original_gt(batch):
    """
    Apply the original ground truth to the batch. Used to calculate the original reward
    """
    for i in range(len(batch)):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["original_gt"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = original_gt

    return batch

# NOTE: Change from ttrl_utils.py/apply_ttrl_gt
def apply_group_confidence_gt(batch, gen_batch_output, n, group_size, tokenizer):
    """
    Apply the ground truth from our method to the batch
    Input:
        batch: DataProto
        gen_batch_output: DataProto
        n: int
        group_size: int
        tokenizer: AutoTokenizer
    Output:
        batch: DataProto
    """
    
    assert len(gen_batch_output) % n == 0, "gen_batch_output length must be divisible by n"
    assert len(gen_batch_output.non_tensor_batch["confs"]) == len(gen_batch_output), "avg_step_confs length must be equal to the number of gen_batch_output"

    num_prompts = len(gen_batch_output) // n
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

    # group_num = len(model_outputs) // group_size
    # (num_prompts, n // group_size), each element is a list of str
    group_gt_list, gt_per_prompt = _batch_group_confidence_vote(model_outputs, n, group_size, confidence)
    print(f"length of group_gt_list: {len(group_gt_list)}, group_size: {group_size}, len of model_outputs: {len(model_outputs)}, len of group_gt_list[0]: {len(group_gt_list[0])}")
    assert len(group_gt_list[0]) == group_size, "group_gt_list length must be equal to the group_size"

    for i in range(num_prompts):
        data_item = batch[i]
        original_gt = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        data_item.non_tensor_batch["reward_model"]["ground_truth"] = np.array(group_gt_list[i])
        data_item.non_tensor_batch["reward_model"]["group_conf_gt"] = np.array(group_gt_list[i])
        data_item.non_tensor_batch["reward_model"]["original_gt"] = original_gt

    # batch.non_tensor_batch["group_conf_gt_list"] = np.array(group_gt_list, dtype=float)
    batch.non_tensor_batch["gt_per_prompt"] = np.array(gt_per_prompt, dtype=int)

    return batch

# NOTE: Change from ttrl_utils.py/_batch_majority_vote
def _batch_group_confidence_vote(model_outputs: List[str], n: int, group_size: int, confidence: List[float]) -> List[List[str]]:
    """
    Used to generate the ground truth by group confidence answer.
    Input:
        model_outputs: list of str, the outputs of the model
        n: int, the number of votes per prompt
        group_size: int, the number of votes per group
        confidence: list of float, the confidence of the outputs
    Output:
        group_gt_list: list of list of str
    """
    assert len(model_outputs) == len(confidence), "model_outputs and confidence must have the same length"
    assert len(model_outputs) % group_size == 0, "model_outputs length must be divisible by group_size"
    
    n_group_total = len(model_outputs) // group_size
    num_prompts = len(model_outputs) // n
    
    group_gt_list = []
    gt_per_prompt = []

    #  _majority_vote_with_confidence
    for i in range(n_group_total):
        group_prompt_outputs = model_outputs[i * group_size:(i + 1) * group_size]
        group_prompt_confidence = confidence[i * group_size:(i + 1) * group_size]
        group_prompt_gt, _ = _majority_vote_with_confidence(group_prompt_outputs, group_prompt_confidence)
        group_gt_list.append(group_prompt_gt)

    assert len(group_gt_list) == n_group_total, "group_gt_list length must be equal to the number of groups"

    # to 2d list
    group_gt_list = [group_gt_list[i * group_size:(i + 1) * group_size] for i in range(num_prompts)]

    assert len(group_gt_list) == num_prompts, "group_gt_list length must be equal to the number of prompts"

    # repear n times 
    group_gt_list = [x for x in group_gt_list for _ in range(n)]

    # count how many different elements in each list
    for i in range(num_prompts):
        gt_per_prompt.append(len(set(group_gt_list[i])))
    
    return group_gt_list, gt_per_prompt

# NOTE: Change from ttrl_utils.py/_majority_vote
def _majority_vote_with_confidence(model_outputs: List[str], confidence_list: List[float]) -> tuple[str, float]:
    """
    Confidence as weight for majority vote, num * confidence.
    Input:
        model_outputs: list of str
        confidence_list: list of float
    Output:
        tuple[str, float]: the most confidence answer and the max score
    """
    assert len(model_outputs) == len(confidence_list), "model_outputs and confidence_list must have the same length"
    assert len(model_outputs) > 0
    model_answers = [extract_answer(generated_text) for generated_text in model_outputs]
    model_answers = [answer for answer in model_answers if answer is not None]
    model_answers = [simplify_expression_string(answer) for answer in model_answers]
    if len(model_answers) == 0:
        return "None", 0.0
    
    # counter = Counter(model_answers)
    score = defaultdict(float)
    for item, conf in zip(model_answers, confidence_list):
        score[item] += conf
    
    max_score = max(score.values())
    most_confidence_answer = max(score, key=score.get)

    return most_confidence_answer, max_score


# === Group Confidence Metrics Computation ===

# NOTE: Change from ttrl_utils.py/compute_ttrl_metrics
def compute_group_confidence_metrics(batch, n, group_size):
    """
    Compute the group confidence metrics.
    """
    assert len(batch) % n == 0, "batch length must be divisible by n"
    num_prompts = len(batch) // n

    # Sort the batch by the ID
    idx = sorted(range(len(batch)), key=lambda x: batch[x].non_tensor_batch["extra_info"]["index"])

    group_conf_reward = []
    gt_reward = []
    group_conf_label = []
    gt_label = []
    # token_level_scores is the reward for the last token of the responsse, for a response with a length of 5 and its reward is 1, its token_level_scores/reward_tensor is [0., 0., 0., 0., 1.], the reward score is save at the last place.
    for i in range(len(batch)):
        group_idx = i % n // group_size # group_idx is the index of the group in the prompt
        data_item = batch[idx[i]]   
        group_conf_reward.append(data_item.batch["token_level_scores"].sum().item())
        gt_reward.append(data_item.batch["token_level_scores_original"].sum().item())
        group_conf_label.append(data_item.non_tensor_batch["reward_model"]["group_conf_gt"][group_idx]) # group_conf_label is the label of the group
        gt_label.append(data_item.non_tensor_batch["reward_model"]["original_gt"])
    
    group_conf_metrics = _batch_compute_group_conf_metrics(group_conf_reward, gt_reward, group_conf_label, gt_label, n, group_size)

    return group_conf_metrics

# NOTE: Change from ttrl_utils.py/_batch_compute_ttrl_metrics
def _batch_compute_group_conf_metrics(
    group_conf_reward: List[float],
    gt_reward: List[float],
    group_conf_label: List[str],
    gt_label: List[str],
    n: int,
    group_size: int,
):
    """
    Calculate the group confidence metrics for batch inputs.
    Input: 
        group_conf_reward: list of float
        gt_reward: list of float
        group_conf_label: list of str
        gt_label: list of str
        n: int
        group_size: int
    Return:
        group_conf_metrics: dict
    """
    assert len(group_conf_reward) == len(gt_reward) == len(group_conf_label) == len(gt_label)
    assert len(group_conf_reward) % n == 0
    n_prompts = len(group_conf_reward) // n
    n_groups_per_prompt = n // group_size
    group_conf_metrics = []

    for i in range(n_prompts):
        prompt_group_conf_reward = group_conf_reward[i * n:(i + 1) * n] # (n, )
        prompt_gt_reward = gt_reward[i * n:(i + 1) * n]
        prompt_group_conf_labels = group_conf_label[i * n:(i + 1) * n] # guessing labels in each group of the prompt
        prompt_gt_labels = gt_label[i * n:(i + 1) * n] # groud truth labels of the prompt

        # check whether the guessing labels in a group is the same
        for j in range(n_groups_per_prompt):
            group_conf_labels = prompt_group_conf_labels[j * group_size:(j + 1) * group_size]
            group_gt_labels = prompt_gt_labels[j * group_size:(j + 1) * group_size]

            assert Counter(group_conf_labels).most_common(1)[0][1] == group_size
            assert Counter(group_gt_labels).most_common(1)[0][1] == group_size

        group_conf_metric = _prompt_compute_group_conf_metrics(prompt_group_conf_reward, prompt_gt_reward, prompt_group_conf_labels, prompt_gt_labels)
        group_conf_metrics.append(group_conf_metric)


    # Compute the average metrics
    group_conf_metrics = {k: sum(d[k] for d in group_conf_metrics) / len(group_conf_metrics) for k in group_conf_metrics[0]}

    return group_conf_metrics

def _prompt_compute_group_conf_metrics(
    group_conf_reward: List[float],
    gt_reward: List[float],
    group_conf_labels: str,
    gt_labels: str,
):
    """
    Calculate the group confidence metrics for prompt inputs.
    Input:
        group_conf_reward: list of float
        gt_reward: list of float
        group_conf_label: list of str
        gt_label: list of str
    Return:
        group_conf_metric: dict
    """
    assert len(group_conf_reward) == len(gt_reward)
    hit_rate = sum([1.0 if grade(i, j) else 0.0 for i, j in zip(group_conf_labels, gt_labels)]) / len(group_conf_labels)

    rewards_hit_rate = 0
    for estimate_reward, true_reward in zip(group_conf_reward, gt_reward):
        if estimate_reward == true_reward:
            rewards_hit_rate += 1
    rewards_hit_rate = rewards_hit_rate / len(group_conf_reward)

    group_conf_metric = {
        "label_accuracy": hit_rate,
        "reward_accuracy": rewards_hit_rate,
        "group_conf_reward": sum(group_conf_reward) / len(group_conf_reward),
        "gt_reward": sum(gt_reward) / len(gt_reward),
        f"pass@{len(group_conf_reward)}": 1.0 if sum(gt_reward) >= 1 else 0.0,
    }
    return group_conf_metric