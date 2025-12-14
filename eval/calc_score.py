# ensure you are using the same llm as a judge for the evaluation

import os
import json
import numpy as np
import multiprocessing
import argparse
from tqdm import tqdm
from openai import OpenAI
import requests

multiprocessing.set_start_method("spawn", force=True)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name", type=str, default="qwen", help="Model name for result save"
)
parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
parser.add_argument(
    "--api_url", type=str, default="http://localhost:18901/v1", help="API URL"
)
parser.add_argument(
    "--vstar_bench_path", type=str, default=None, help="Path to the V* benchmark"
)
parser.add_argument(
    "--save_path", type=str, default=None, help="Path to save the results"
)
parser.add_argument(
    "--eval_model_name", type=str, default=None, help="Model name for evaluation"
)
parser.add_argument("--num_workers", type=int, default=8)
args = parser.parse_args()

openai_api_key = args.api_key
openai_api_base = args.api_url
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.eval_model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    eval_model_name = models["data"][0]["id"]
else:
    eval_model_name = args.eval_model_name

vstar_bench_path = args.vstar_bench_path
test_types = ["direct_attributes", "relative_position"]
per_type_acc = {}
for test_type in test_types:
    per_type_acc[test_type] = []
all_acc = []
all_function_call_count = []
abc_map = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judgement is 1; if they are different, Judgement is 0. Just output Judgement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: A. The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: A. The barrier is on the left side of the picture.
[Model_answer] : A
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: A. Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: A. No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: A. The boy is wearing pants.
[Model_answer] : C. The girl in the picture is wearing pants.
Judgement: 0
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: A. Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: A. The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


result_root_path = args.save_path
result_root_path = os.path.join(result_root_path, args.model_name)
all_acc = []
per_type_acc = {}
error_nums_type = {}
for test_type in test_types:
    per_type_acc[test_type] = []
    error_nums_type[test_type] = 0


def process(line):
    # breakpoint()
    line = line.strip()
    data = json.loads(line)
    question = data["question"]
    answer = data["answer"]
    pred_ans = data["pred_ans"]
    # pred_output = data["pred_output"]
    answer = "A. " + answer
    status = data["status"]
    function_call_count = data["function_call_count"]

    if "\\boxed" in pred_ans:
        pred_ans = pred_ans.split("\\boxed{")[1].split("}")[0]

    # rule base check
    acc_reward = 0.0
    if len(pred_ans) == 1:
        if pred_ans == "A":
            acc_reward = 1.0
        else:
            acc_reward = 0.0
    elif len(pred_ans) == 2 and "." in pred_ans:
        if "A" in pred_ans:
            acc_reward = 1.0
        else:
            acc_reward = 0.0
    elif answer in pred_ans:
        acc_reward = 1.0
    else:
        full_prompt = get_prompt(pred_ans, answer, question)

        chat_response = client.chat.completions.create(
            model=eval_model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.0,
        )
        response = chat_response.choices[0].message.content.strip()
        if "ERROR" in pred_ans or "error" in status:
            error_nums_type[test_type] += 1

        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
    
    # # The following is not used and therefore commented out to pass ruff linting.
    # for p_out in pred_output:
    #     if p_out["role"] == "system" or p_out["role"] == "user":
    #         continue
    #     p_content = p_out["content"]
    #     if p_content is str:
    #         p_content_msg = p_content.strip()
    #     elif p_content is list:
    #         for _p_content in p_content:
    #             if _p_content["type"] == "text":
    #                 p_content_msg = _p_content["text"]

    return acc_reward, data


if __name__ == "__main__":
    error_preds = []
    
    # breakpoint()

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        result_path = os.path.join(result_root_path, save_name)
        save_json = []
        pool = multiprocessing.Pool(processes=args.num_workers)
        with open(result_path, "r") as f:
            lines = f.readlines()

        with tqdm(total=len(lines), desc="Judging V* " + test_type) as pbar:
            for result in pool.imap(process, lines):
            # for line in lines:
                # breakpoint()
                # result = process(line)
                if result is not None:
                    acc_reward, data = result
                    acc = acc_reward
                    all_acc.append(acc)
                    all_function_call_count.append(data.get("function_call_count", 0))
                    per_type_acc[test_type].append(acc)
                    if acc_reward != 1.0:
                        error_preds.append(
                            {
                                "pred_ans": data["pred_ans"],
                                "question": data["question"],
                                "answer": data["answer"],
                            }
                        )

                    data["acc"] = acc

                    save_json.append(data)
                    pbar.update(1)

        pool.close()
        pool.join()

        with open(
            os.path.join(result_root_path, save_name.replace(".jsonl", "_acc.jsonl")),
            "w",
        ) as f:
            for item in save_json:
                f.write(json.dumps(item) + "\n")

    final_acc = {}
    for test_type in test_types:
        print(
            f"Accuracy for {test_type}: {np.mean(per_type_acc[test_type]) * 100:.2f}%"
        )
        final_acc[test_type] = np.mean(per_type_acc[test_type]) * 100
    print(f"Overall Accuracy: {np.mean(all_acc) * 100:.2f}%")
    print(f"Overall Function Call Count: {np.mean(all_function_call_count)}")
    final_acc["overall"] = np.mean(all_acc) * 100
    # print(error_nums_type)
    print(f"error_nums_type: {error_nums_type}")
    final_acc["error_nums"] = error_nums_type
    final_acc["error_preds"] = error_preds
    with open(os.path.join(result_root_path, "final_acc.json"), "w") as f:
        json.dump(final_acc, f, indent=4)
