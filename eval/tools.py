from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_images_from_messages
from qwen_agent.llm.schema import ContentItem
from qwen_vl_utils import smart_resize

from openai import OpenAI

from typing import Dict, Union
import os
import requests
from PIL import Image, ImageDraw
from io import BytesIO
import base64

import ast

# still use the qwen_agent tools
# if a better framework is found, maybe transfer to the new framework


# ------------------------------------------------------------
# Qwen Constants
# ------------------------------------------------------------
FACTOR = 28
MIN_PIXELS = 4 * FACTOR * FACTOR
MAX_PIXELS = 16384 * FACTOR * FACTOR


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def encode_pil_image_to_base64(image: Image.Image) -> str:
    assert isinstance(image, Image.Image), "Image must be a PIL image"
    buffer = BytesIO()
    image.save(buffer, format="PNG") # should not affect whether the org image is jpeg or png
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def image_preprocessing(image, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if isinstance(image, str):
        image = Image.open(image)
    
    assert isinstance(image, Image.Image), "Image must be a PIL image"
    
    width, height = image.size
    input_width, input_height = smart_resize(width, height, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    image = image.resize((input_width, input_height))
    return image

VLM_SUBAGENT_DESC = """
USE THIS SUBAGENT TOOL TO SOLVE SUBTASKS TO HELP YOU COMPLETE THE TASK.

This tool is a vision-language model (VLM) model. 
You can make it into any kind of tool on downstream tasks of VLM by carefully designing the prompt. 
Possible usages include OCR, Caption, Reasoning, etc.
"""
    
@register_tool('vlm_subagent_tool')
class QwenImageVLMTool(BaseToolWithFileAccess):
    description = VLM_SUBAGENT_DESC
    parameters = {
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
                'description': 'The type of the task you want to perform. For example, "subregion_caption", "subregion_ocr", and "subregion_question_answering", etc.'
            },
            'bbox_2d': {
                'type': 'array',
                'items': {
                    'type': 'number'
                },
                'minItems': 4,
                'maxItems': 4,
                'description': 'The bounding box of the region if you want to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.'
            }
        },
        'required': ['prompt', 'img_idx', 'task_type', 'bbox_2d']
    }
    
    def register_api(self, cfg: Dict):
        assert "api_url" in cfg, "api_url must be in cfg, should be like the base url of the oai api"
        self.api_url = cfg["api_url"]
        
        assert "model_id" in cfg, "model_id must be in cfg, like `Qwen2.5-VL-7B-Instruct`"
        self.model_id = cfg["model_id"]
        self.api_key = cfg.get("api_key", None)
        
    def inference_with_oai(
        self, 
        image_url, 
        prompt, 
        system_prompt: str | None = None, 
        min_pixels=MIN_PIXELS, 
        max_pixels=MAX_PIXELS
    ) -> str:
        
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_url
        )
        
        system_prompt = system_prompt or "You are a helpful assistant."
        
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "image_url": {"url": image_url}
                    },
                    {
                        "type": "text",
                        "text": "This is a cropped region of the original image. Try to solve the subtask if possible and return the answer. Otherwise, ask the user to make another call." + prompt
                    }
                ]
            }
        ]
        
        # do not change the default parameters if not necessary!!!
        completion = client.chat.completions.create(
            model = self.model_id,
            messages = messages,
            temperature=1e-6,
            top_p=1.0,
            max_completion_tokens=1024,
            extra_body={
                "repetition_penalty": 1.1,
                "top_k": 50,
                "length_penalty": 1.0,
                # "diversity_penalty": 0.0,
                # "typical_p": 1.0
            }
        )
        
        return completion.choices[0].message.content
        
    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        
        img_idx = params['img_idx']
        images: list[str] = extract_images_from_messages(kwargs.get('messages', []))
        prompt = params['prompt']
        task_type = params['task_type']
        prompt = f"[{task_type}] {prompt}"
        bbox_2d = params.get('bbox_2d', None)
        
        try:
            # open image, currently only support the first image
            image_arg = images[img_idx]
            if os.path.exists(image_arg):
                # is simply a local file
                image = Image.open(image_arg)
            elif image_arg.startswith('file://'):
                # is a local file but in url
                image = Image.open(image_arg[len('file://'):])
            elif image_arg.startswith('http'):
                # is a url
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif image_arg.startswith('data:image/'):
                # is a base64 encoded image
                image_base64 = image_arg.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_base64)))
            else:
                raise ValueError(f"Invalid image argument: {image_arg[:100]}")
        except Exception as e:
            return f"Error: Invalid input image grounding params {params}\n" + f"Error: {e}"

        width, height = image.size
        input_width, input_height = smart_resize(
            width, height, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        if (input_width, input_height) != (width, height):
            image = image.resize((input_width, input_height))
            
        draw = ImageDraw.Draw(image)
            
        # crop the image if bbox_2d is provided
        if bbox_2d is not None:
            left, top, right, bottom = bbox_2d
            
            whole_bbox_2d = [0, 0, input_width, input_height]
            # draw_interpolation_factor = 0.02
            # draw_bbox_2d = [int(draw_interpolation_factor * x + (1 - draw_interpolation_factor) * y) 
            #                 for (x, y) in zip(whole_bbox_2d, bbox_2d)]
            # left, top, right, bottom = draw_bbox_2d
            # draw.rectangle([left, top, right, bottom], outline=(255, 0, 0, 80), width=2)
            
            
            crop_interpolation_factor = 0.02
            crop_bbox_2d = [int(crop_interpolation_factor * x + (1 - crop_interpolation_factor) * y) 
                            for (x, y) in zip(whole_bbox_2d, bbox_2d)]
            
            left, top, right, bottom = crop_bbox_2d
            
            cropped_image = image.crop((left, top, right, bottom))
            image_url = f"data:image/jpeg;base64,{encode_pil_image_to_base64(cropped_image)}"
        else:
            image_url = f"data:image/jpeg;base64,{encode_pil_image_to_base64(image)}"
            
            
        if bbox_2d is None:
            return [ContentItem(text=f"Error: bbox_2d is required for this subregion task: {task_type}. Please try to call subagent again with a valid bbox_2d.")]
            
        response = self.inference_with_oai(
            image_url, prompt, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        
        if bbox_2d is not None:
            output_content = [
                ContentItem(image=image_url),
                ContentItem(text=f"VLM Prompt: {prompt}"),
                ContentItem(text=f"Queried at Bounding Box {bbox_2d} of the original image."),
                ContentItem(text=f"VLM Result: \n```\n{response}\n```"),
            ]
        else:
            output_content = [
                ContentItem(text=f"VLM Prompt: {prompt}"),
                ContentItem(text=f"VLM Result: \n```\n{response}\n```"),
            ]
        
        return output_content
    
    def parse_json(self, response: str) -> list[dict]:
        lines = response.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                # remove everything before "```json"
                json_output = "\n".join(lines[i+1:])
                json_output = json_output.split("```")[0]
                break
        
        
        # ref: https://github.com/QwenLM/Qwen3-VL/blob/2f25a646fb0f329647428eb8dacf19293de6f5d4/cookbooks/spatial_understanding.ipynb
        try: 
            json_output = ast.literal_eval(json_output)
        except Exception as e:
            end_idx = json_output.rfind('"}') + len('"}')
            truncated_text = json_output[:end_idx] + "]"
            json_output = ast.literal_eval(truncated_text)
            
        return json_output
    

@register_tool('qwen_vl_25_zoom_in_tool')
class QwenVL25ZoomInTool(BaseToolWithFileAccess):
    description = 'Use this tool to zoom in on a specific region of an image by cropping it based on a bounding box.'
    parameters = {
        'type': 'object',
        'properties': {
            'bbox_2d': {
                'type': 'array',
                'items': {
                    'type': 'number'
                },
                'minItems': 4,
                'maxItems': 4,
                'description': 'The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.'
            },
            'img_idx': {
                'type': 'number',
                'description': 'The index of the image (starting from 0) in the messages to be zoomed in.'
            }
        },
        'required': ['bbox_2d', 'img_idx']
    }
    
    
    def call(self, params: Union[str, dict], **kwargs) -> str:
        params = self._verify_json_format_args(params)
        
        img_idx = params['img_idx']
        images: list[str] = extract_images_from_messages(kwargs.get('messages', []))
        bbox_2d = params['bbox_2d']
        
        try:
            # open image, currently only support the first image
            image_arg = images[img_idx]
            if os.path.exists(image_arg):
                # is simply a local file
                image = Image.open(image_arg)
            elif image_arg.startswith('file://'):
                # is a local file but in url
                image = Image.open(image_arg[len('file://'):])
            elif image_arg.startswith('http'):
                # is a url
                response = requests.get(image_arg)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            elif image_arg.startswith('data:image/'):
                # is a base64 encoded image
                image_base64 = image_arg.split(',')[1]
                image = Image.open(BytesIO(base64.b64decode(image_base64)))
            else:
                raise ValueError(f"Invalid image argument: {image_arg[:100]}")
        except Exception as e:
            return f"Error: Invalid input image grounding params {params}\n" + f"Error: {e}"
        
        width, height = image.size
        input_width, input_height = smart_resize(
            width, height, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        if (input_width, input_height) != (width, height):
            image = image.resize((input_width, input_height))
        
        cropped_image = self.crop_image(image, bbox_2d, input_height, input_width)
        
        return [ContentItem(image=f"data:image/jpeg;base64,{encode_pil_image_to_base64(cropped_image)}")]
    
    def crop_image(self, image: Image.Image, bbox_2d: list[int], input_height: int, input_width: int) -> Image.Image:
        width, height = image.size
        
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bbox_2d[1]/input_height * height)
        abs_x1 = int(bbox_2d[0]/input_width * width)
        abs_y2 = int(bbox_2d[3]/input_height * height)
        abs_x2 = int(bbox_2d[2]/input_width * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1
        
        cropped_image = image.crop((abs_x1, abs_y1, abs_x2, abs_y2))
        cropped_width, cropped_height = cropped_image.size
        cropped_width, cropped_height = smart_resize(
            cropped_width, cropped_height, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS
        )
        cropped_image = cropped_image.resize((cropped_width, cropped_height))
        
        return cropped_image