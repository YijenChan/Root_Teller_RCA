import json
import json5
import os
import re
import traceback

import numpy as np
import torch

all_possibilities = {'node-3', 'node-2', 'emailservice-1', 'adservice2-0', 'checkoutservice2-0',
                     'recommendationservice-1', 'adservice', 'node-5', 'paymentservice-0', 'frontend-0', 'frontend-1',
                     'cartservice-2', 'emailservice', 'cartservice2-0', 'shippingservice-1', 'productcatalogservice',
                     'adservice-http', 'shippingservice2-0', 'frontend-http', 'currencyservice', 'cartservice',
                     'recommendationservice-2', 'paymentservice2-0', 'shippingservice-2', 'cartservice-0',
                     'emailservice-0', 'frontend-2', 'checkoutservice', 'currencyservice-1', 'adservice-2',
                     'paymentservice-2', 'node-6', 'checkoutservice-1', 'checkoutservice-0', 'currencyservice-2',
                     'recommendationservice', 'redis-cart2-0', 'cartservice-1', 'shippingservice',
                     'productcatalogservice-1', 'adservice-1', 'adservice-0', 'paymentservice-1',
                     'productcatalogservice-2', 'currencyservice-0', 'currencyservice2-0', 'node-4', 'paymentservice',
                     'emailservice-2', 'frontend2-0', 'recommendationservice2-0', 'productcatalogservice2-0',
                     'emailservice2-0', 'checkoutservice-2', 'productcatalogservice-0', 'node-1',
                     'recommendationservice-0', 'redis-cart-0', 'shippingservice-0'}


def get_tool_res_from_content(content):
    # Match tool blocks with non-greedy regular expressions.
    pattern = re.compile(
        r'<\|eot_id\|>\s*<\|start_header_id\|>tool<\|end_header_id\|>.*?<\|eot_id\|>'  # Pattern 1
        r'|'  # Alternative
        r'<\|im_start\|>user\s*<tool_response>.*?</tool_response>\s*<\|im_end\|>'  # Pattern 2
        r'|'
        r'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>.*?<｜tool▁output▁end｜><｜tool▁outputs▁end｜>'
        r'|'
        r'<\|eot_id\|><\|start_header_id\|>ipython<\|end_header_id\|>.*?<\|eot_id\|>',
        re.DOTALL
    )

    # Extract all tool blocks.
    tool_blocks = pattern.findall(content)
    return " ".join(tool_blocks)


def robust_json_loads(json_str):
    try:
        return json.loads(json_str), 1.0
    except json.JSONDecodeError as e:
        fix_attempts = [
            # Repair extra or missing braces.
            lambda s: s.replace('}}', '}'),
            lambda s: s.replace('"{', '{').replace('}"', '}'),
            lambda s: s + '}' if s.count('{') > s.count('}') else s,
            lambda s: '{' + s if s.count('}') > s.count('{') else s,

            # Repair malformed keys and values.
            lambda s: re.sub(r'(\w+):', r'"\1":', s),
            lambda s: s.replace('""', '"'),
            lambda s: s.replace("'", '"'),
            lambda s: s.replace('\\"', '"'),

            # Repair array and object structure.
            lambda s: re.sub(r',\s*}', '}', s),
            lambda s: re.sub(r'\],\s*\]', '\]', s),
            lambda s: s.replace('}{', '},{'),

            # Handle special malformed cases.
            lambda s: s.replace('true', '"true"').replace('false', '"false"').replace('null', '"null"'),
            lambda s: re.sub(r'\s*=\s*', ': ', s),
        ]

        for attempt in fix_attempts:
            fixed_str = attempt(json_str)
            try:
                return json.loads(fixed_str), 0.8  # Reward a successful repair.
            except:
                try:
                    return json5.loads(fixed_str), 0.8
                except:
                    continue

        return {}, 0.2  # Base score for an irreparable response.
    except Exception as e:
        return {}, 0.0  # Unrecoverable error.


def hierarchical_format_score(json_data):
    score = 0  # Initial score.

    # Required fields.
    if json_data.get("name", "") == "print_results":
        score += 0.2
    if "arguments" in json_data:
        score += 0.2

    # Data completeness.
    arguments_str = json_data.get("arguments", "[]")
    if type(arguments_str) == str:
        arguments_block, arguments_score = robust_json_loads(json_data.get("arguments", "[]"))
        score += 0.2 * arguments_score
    else:
        arguments_block = arguments_str
        score += 0.2
    if arguments_block:
        root_causes = arguments_block.get("root_causes", [])
        if len(root_causes) < 3:
            score += 0.4 * (len(root_causes) / 3)  # Proportional score.
        else:
            score += 0.4
        for item in root_causes:
            if not all(k in ["node", "service", "pod"] for k in item.keys()):
                score -= 0.1  # Penalize each invalid field.
    return score


def get_result_from_content(content):
    tools = []
    # Match tool-call tag content with a non-greedy expression.
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    # Extract all tool-call blocks.
    tool_call_blocks = pattern.findall(content)

    total_score = 0
    for block in tool_call_blocks:
        try:
            # Strip whitespace and parse JSON.
            cleaned_block = block.strip()
            tool_data, fix_score = robust_json_loads(cleaned_block)
            if tool_data:
                hierarchical_score = hierarchical_format_score(tool_data)
                total_score += fix_score * hierarchical_score

                # Validate required fields.
                name = tool_data.get("name", "")
                arguments_str = tool_data.get("arguments", "[]")
                if type(arguments_str) == str:
                    arguments_json, _ = robust_json_loads(tool_data.get("arguments", "[]"))
                else:
                    arguments_json = arguments_str

                tools.append({
                    "function": {
                        "name": name,
                        "arguments": arguments_json
                    }
                })
                total_score = total_score / 10
            else:
                total_score = 0.05
        except Exception as e:
            traceback.print_exc()
            return [], total_score

    results = []
    for tool in tools:
        if "root_causes" in tool["function"]["arguments"]:
            root_causes = tool["function"]["arguments"]["root_causes"]
            for root_cause in root_causes:
                for key, value in root_cause.items():
                    if key in ["service", "pod", "node"]:
                        results.append(value.lower())
    return results, total_score


def count_unique_tool_calls(content):
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>")
    tool_call_blocks = pattern.findall(content)
    unique_tool_call_blocks = set(tool_call_blocks)
    return len(unique_tool_call_blocks)


def reward_func(queries, prompts, labels, save_path):
    ground_truth_path = os.environ.get("THINKFL_GROUND_TRUTH", "all_groundtruth.json")
    with open(ground_truth_path, "r") as f:
        all_groundtruth = json.loads(f.read())
        # queries is prompts + responses
        queries = queries[0]
        prompts = prompts[0]
        response = queries[len(prompts):]
        # tool_results = get_tool_res_from_content(response).lower()
        tool_call_num = count_unique_tool_calls(response)
        if tool_call_num <= 0:
            return torch.tensor([0], dtype=torch.float32)
        last_call = response[response.rfind('<tool_call>'):]
        results = None
        ground_truth = None
        target_timestamp = None
        content_score = 0
        try:
            for part in prompts.split():
                if len(part) == 13 and part.startswith('16'):
                    target_timestamp = part
            ground_truth = all_groundtruth[target_timestamp]
            label = ground_truth['cmdb_id'].lower()

            results, format_score = get_result_from_content(last_call)
            if len(results) == 0:
                content_score = 0
            else:
                if label in results:
                    label_index = results.index(label)
                    content_score = 1.0 / (label_index + 1)
                # for result in results:
                #     if result not in all_possibilities:
                #         content_score -= 0.1

        except Exception as e:
            traceback.print_exc()
        with open(save_path, "a+") as f:
            f.write("=" * 30 + "reward_result" + "=" * 30 + "\n")
            f.write(str(last_call) + "\n\n")
            f.write(str(results) + "\n\n")
            f.write("tool_call_num=" + str(tool_call_num) + "\n\n")
            f.write(str(target_timestamp) + "\n\n")
            f.write(str(ground_truth) + "\n\n")
            f.write("score=" + str(content_score) + "\n")
            f.write(str(response) + "\n\n")
            f.write("=" * 30 + "\n")
        return torch.tensor([content_score], dtype=torch.float32)

# def reward_func(queries, prompts, labels, save_path):
#     with open(os.environ.get("THINKFL_GROUND_TRUTH", "all_groundtruth.json"), "r") as f:
#         all_groundtruth = json.loads(f.read())
#         # queries is prompts + responses
#         queries = queries[0]
#         prompts = prompts[0]
#         response = queries[len(prompts):]
#         tool_results = get_tool_res_from_content(response).lower()
#         tool_call_num = response.count('<tool_call>')
#         last_call = response[response.rfind('<tool_call>'):]
#         results = None
#         ground_truth = None
#         try:
#             results = get_result_from_content(last_call)
#             for part in prompts.split():
#                 if len(part) == 13 and part.startswith('16'):
#                     target_timestamp = part
#             ground_truth = all_groundtruth[target_timestamp]
#             label = ground_truth['cmdb_id'].lower()
#             if len(results) == 0:
#                 score = 0
#             else:
#                 is_hallucination = False
#                 score = 0.5
#                 for result in results:
#                     if result not in tool_results:
#                         score -= 0.05
#                         is_hallucination = True
#                 if not is_hallucination:
#                     if len(results) == 1:
#                         score = 0.6
#                     elif len(results) == 2:
#                         score = 0.8
#                     else:
#                         if label in results:
#                             label_index = results.index(label)
#                             score = 1.5 + 1.0 / (label_index + 1)
#                         else:
#                             score = 1 + 1.0 / 50 * tool_call_num
#         except Exception as e:
#             traceback.print_exc()
#             score = 0
#         with open(save_path, "a+") as f:
#             f.write("=" * 30 + "reward_result" + "=" * 30 + "\n")
#             f.write(str(last_call) + "\n\n")
#             f.write(str(results) + "\n\n")
#             f.write(str(tool_call_num) + "\n\n")
#             f.write(str(target_timestamp) + "\n\n")
#             f.write(str(ground_truth) + "\n\n")
#             f.write("score=" + str(score) + "\n")
#             f.write(str(response) + "\n\n")
#             f.write("=" * 30 + "\n")
#         return torch.tensor([score], dtype=torch.float32)
