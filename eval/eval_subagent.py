import os
import json
import numpy as np
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
from tqdm import tqdm
import math
from io import BytesIO
from PIL import Image
import base64
from openai import OpenAI
import requests
import re


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='Qwen2.5-VL-7B-Instruct', help='Model name for result save')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://localhost:18901/v1', help='API URL')
parser.add_argument('--vstar_bench_path', type=str, default=None, help='Path to the V* benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()


openai_api_key = args.api_key
openai_api_base = args.api_url

TOOL_NAME = "vlm_subagent_tool"
TOOL_PARAMS = {
    'type': 'object',
    'properties': {
        'prompt': {
            'type': 'string',
            'description': "The prompt to be passed to the VLM model. Wisely design the prompt, so you can make it into any kind oftool on downstream tasks of VLM."
        },
        'img_idx': {
            'type': 'number',
            'description': 'The index of the image (starting from 0) in the messages to be analyzed.'
        },
        'task_type': {
            'type': 'string',
            'description': 'The type of the task you want to perform. For example, "full_ocr", "grounding", "subregion_caption", "subregion_ocr", and "subregion_question_answering". If this is a subregion task, you must provide `bbox_2d` parameter.'
        },
        'bbox_2d': {
            'type': 'array',
            'items': {
                'type': 'number'
            },
            'minItems': 4,
            'maxItems': 4,
            'description': 'The bounding box of the region if you want to zoom in, as [x1, y1, x2, y2] (left, top, right, bottom). If you want to perform a subregion task, you must provide `bbox_2d`.'
        }
    },
    'required': ['prompt', 'img_idx', 'task_type']
}
TOOL_DESC = """
USE THIS SUBAGENT TOOL TO SOLVE SUBTASKS TO HELP YOU COMPLETE THE TASK.

This tool is a vision-language model (VLM) model. 
You can make it into any kind of tool on downstream tasks of VLM by carefully designing the prompt. 
Possible usages include OCR, Caption, Reasoning, etc.
"""
TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESC,
        "parameters": TOOL_PARAMS,
    }
}
TOOL_STRING = f"<tools>\n{json.dumps(TOOL_SCHEMA)}\n</tools>"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
if args.model_name is None:
    response = requests.get(f"{openai_api_base}/models")
    models = response.json()
    model_name = models['data'][0]['id']
else:
    model_name = args.model_name

vstar_bench_path = args.vstar_bench_path
save_path = args.save_path
save_path = os.path.join(save_path, args.model_name)
os.makedirs(save_path, exist_ok=True)
abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28

instruction_prompt_system = f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
{TOOL_STRING}

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>

## Example Usages
1. Full Image OCR: Output only the text content from the image without any additional descriptions or formatting.
    ```json
        <tool_call>
        {{
            "name": "vlm_subagent_tool",
            "arguments": {{
                "prompt": "Please output only the text content from the image without any additional descriptions or formatting.",
                "img_idx": 0,
                "task_type": "ocr",
            }}
        }}
        </tool_call>
    ```
    - Use it when you want to know all the texts in the image
2. Visual Grounding: Locate the objects in the image and return the bbox_2d.
    - Usage: Use it when you want to find the hardly visible objects in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Outline the position of each object and output all the coordinates in JSON format.",
                    "img_idx": 0,
                    "task_type": "grounding",
                }}
            }}
            </tool_call>
        ```
3. Subregion Caption: Describe the image in a few sentences. Must provide `bbox_2d` to zoom in on the region.
    - Usage: Use it when you want to take a close look at a specific region in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Please describe the image in a few sentences.",
                    "img_idx": 0,
                    "task_type": "subregion_caption",
                    "bbox_2d": [63, 32, 150, 170]
                }}
            }}
            </tool_call>
        ```
4. Subregion OCR: Read out the text in the region located by the bbox_2d. Must provide `bbox_2d` to zoom in on the region.
    - Usage: Use it when you want to know the text in a specific region in the image.
        ```json
            <tool_call>
            {{
                "name": "vlm_subagent_tool",
                "arguments": {{
                    "prompt": "Outline the position of each text and output all the coordinates in JSON format.",
                    "img_idx": 0,
                    "task_type": "subregion_ocr",
                    "bbox_2d": [63, 32, 150, 170]
                }}
            }}
            </tool_call>
        ```
"""

USER_PROMPT_V2 = "\nCall **vlm_subagent_tool** if needed, then answer. Format strictly as: <tool_call>...</tool_call> (if tools needed)  <answer>...</answer> "

instruction_prompt_before = """Question: {question}
Options: {options}
""" + USER_PROMPT_V2

user_prompt = USER_PROMPT_V2

start_token = "<tool_call>"
end_token = "</tool_call>"


# noqa: add a tool parser for the prompt parser so we could excape from latex parsing hell
def backup_tool_parser(tool_call_str: str) -> tuple[str]:
    pattern = r'"prompt"\s*:\s*"([^"]*?(?:\\.[^"]*?)*)"'
    
    # extract the prompt
    prompt = re.findall(pattern, tool_call_str, re.DOTALL)
    assert len(prompt) == 1, f"Expected 1 prompt, got {len(prompt)}: {prompt}"
    prompt = prompt[0]
    # replace the prompt with ""
    cleaned_text = re.sub(pattern, '"prompt": ""', tool_call_str, re.DOTALL)
    return prompt, cleaned_text


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

# the following code is copied from qwen-vl-utils
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def process(img_arg):
    img, test_path = img_arg
    img_path = os.path.join(test_path, img)
    anno_path = os.path.join(test_path, img.replace('.jpg', '.json'))
    with open(anno_path, 'r') as f:
        anno = json.load(f)
    question = anno['question']
    options = anno['options']

    option_str = "\n"
    for i in range(len(options)):
        option_str += abc_map[i + 1] + '. ' + options[i] + '\n'
    
    prompt = instruction_prompt_before.format(question=question, options=option_str)
    pil_img = Image.open(img_path)
    ori_width, ori_height = pil_img.size
    resize_w, resize_h = smart_resize(ori_width, ori_height, factor=IMAGE_FACTOR)
    img = pil_img.resize((resize_w, resize_h), resample=Image.BICUBIC)

    # base64_image = encode_image_to_base64(img_path)
    base64_image = encode_pil_image_to_base64(img)

    messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    print_messages = [
        {
            "role": "system",
            "content": instruction_prompt_system,
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"}},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    chat_message = messages

    response_message = ""

    status = 'success'
    try_count = 0
    turn_idx = 0
    function_call_count = 0
    try:
        while '</answer>' not in response_message:
            # print(f"try_count: {try_count}")
            if '</answer>' in response_message and '<answer>' in response_message:
                break

            if try_count > 10:
                break

            params = {
                "model": model_name,
                "messages": chat_message,
                "temperature": 0.0,
                "max_tokens": 10240,
                "stop": ["<|im_end|>\n".strip(), "</tool_call>"],
            }
            response = client.chat.completions.create(**params)
            response_message = response.choices[0].message.content
            
            if start_token in response_message:
                # remove 'addCriterion' from the response_message
                # see https://docs.unsloth.ai/new/vision-reinforcement-learning-vlm-rl#qwen-2.5-vl-vision-rl-issues-and-quirks
                response_message = response_message.replace('addCriterion', '')
                function_call_count += 1
                action_list = response_message.split(start_token)[1].split(end_token)[0].strip()
                try:
                    action_list = eval(action_list)
                except Exception as e:
                    try:
                        prompt, cleaned_text = backup_tool_parser(action_list)
                        action_list = eval(cleaned_text)
                        action_list['arguments']['prompt'] = prompt
                    except Exception as e:
                        # print(f"Error Parsing Tool Call: {e} {action_list=}")
                        raise Exception(f"Error Parsing Tool Call: {e} {action_list=}")

                bbox_list = []
                image_content_list = []
                
                arguments = action_list['arguments']
                task_type = arguments['task_type']
                prompt = arguments['prompt']
                bbox_2d = arguments.get('bbox_2d', None)
                if bbox_2d is not None:
                    bbox = bbox_2d
                    # left, top, right, bottom = bbox
                    
                    # interplolation
                    interpolation_factor = 0.3
                    whole_bbox_2d = [0, 0, resize_w, resize_h]
                    new_bbox = [int(interpolation_factor * x + (1 - interpolation_factor) * y) for (x, y) in zip(whole_bbox_2d, bbox_2d)]
                    
                    left, top, right, bottom = new_bbox
                    # FIXME: the original image is resized, so the bbox_2d is not the same as the original image.
                    cropped_image = img.crop((left, top, right, bottom))
                    new_w, new_h = smart_resize((right - left), (bottom - top), factor=IMAGE_FACTOR)
                    cropped_image = cropped_image.resize((new_w, new_h), resample=Image.BICUBIC)
                    cropped_pil_image = encode_pil_image_to_base64(cropped_image)
                    bbox_list.append(bbox)
                    cropped_pil_image_content = {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cropped_pil_image}"}}
                    image_content_list.append(cropped_pil_image_content)
                else:
                    image_content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

                if len(bbox_list) == 1:
                    bbox_list = bbox_list[0]
                user_msg = user_prompt

                # get the content from the subagent
                subagent_input_messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "function",
                        "content": image_content_list + [
                            {"type": "text", "text": f"[{task_type}] {prompt}"},
                        ],
                    }
                ]
                
                subagent_call_params = {
                    "model": model_name,
                    "messages": subagent_input_messages,
                    "temperature": 0.0,
                    "max_tokens": 10240,
                    "stop": ["<|im_end|>\n".strip()],
                }
                subagent_response = client.chat.completions.create(**subagent_call_params)
                subagent_response_content: str = subagent_response.choices[0].message.content
                
                content_f = []
                content_f.append({"type": "text", "text": "<tool_response>"})
                content_f.append({"type": "text", "text": f"[Queried at Bounding Box {bbox_2d} of the original image.]"})
                content_f.append({"type": "text", "text": subagent_response_content})
                content_f.append({"type": "text", "text": "</tool_response>"})

                _message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    },
                    {
                        "role": "function",
                        "content": content_f,
                    },
                    # {
                    #     "role": "user",
                    #     "content": [
                    #         {"type": "text", "text": user_msg},
                    #     ],
                    # }
                ]

                chat_message.extend(_message)
                p_message = _message
            
                # p_message =[
                #     {
                #         "role": "assistant",
                #         "content": response_message,
                #     },
                #     {
                #         "role": "user",
                #         "content": [
                #             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,"}},
                #             {"type": "text", "text": user_msg},
                #         ],
                #     }
                # ]
                print_messages.extend(p_message)
                turn_idx += 1
            else:
                p_message =[
                    {
                        "role": "assistant",
                        "content": response_message,
                    }
                ]
                print_messages.extend(p_message)


            try_count += 1
    except Exception as e:
        print(f"Error!!!!", e)
        status = 'error'
                

    if '</answer>' in response_message and '<answer>' in response_message:
        output_text = response_message.split('<answer>')[1].split('</answer>')[0].strip()
    else:
        output_text = response_message

    save_info = {}
    save_info['image'] = base64_image
    save_info['question'] = question
    save_info['answer'] = anno['options'][0]
    save_info['pred_ans'] = output_text
    save_info['pred_output'] = print_messages
    save_info['status'] = status
    save_info['function_call_count'] = function_call_count
    return save_info


if __name__ == "__main__":
    test_types = ['direct_attributes', 'relative_position']

    for test_type in test_types:
        save_name = f"result_{test_type}_{args.model_name}.jsonl"
        save_json = []
        test_path = os.path.join(vstar_bench_path, test_type)
        pool = multiprocessing.Pool(processes=args.num_workers)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
        image_args = [[img, test_path] for img in image_files]

        with tqdm(total=len(image_args), desc="Processing V* "+test_type) as pbar:
            for result in pool.imap(process, image_args):
                if result is not None:
                    save_json.append(result)
                    pbar.update(1)

        pool.close()
        pool.join()
        
        # with tqdm(total=len(image_args), desc="Processing V* "+test_type) as pbar:
        #     for result in image_args:
        #         result = process(result)
        #         if result is not None:
        #             save_json.append(result)
        #             pbar.update(1)
    
        with open(os.path.join(save_path, save_name), 'w') as f:
            for item in save_json:
                f.write(json.dumps(item) + '\n')