# Placeholders
# openai_api_key = 
# gcp_project_id = 
# claude_api_key = 
# qwen_api_key=
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 


#default model is GPT_model="gpt-4o-2024-05-13"
import base64
import time
import pickle
import os
import uuid
import pandas as pd
from tqdm import tqdm
import traceback
import random
from PIL import Image
import base64
from PIL import Image
import io
from openai import OpenAI

import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models
import requests
import time
import base64
from PIL import Image
import io
import os
import asyncio

import anthropic

class InternVLAPI:
    def __init__(
        self,
        api_url="http://101.132.98.120:11004/chat/",
        api_key="InternVL-2-Pro_256f846c96d0b93ac3b46a87d3b5f0f91f7316b8aa7e454466beb585cf1f3449_UmPCEbPZ",
        img_token="<<IMG>>",
    ):
        """
        Class for API calls to InterVL model

        api_url[str]: the URL for the API endpoint
        api_key[str]: the API key for authentication
        img_token[str]: string to be replaced with images
        """
        self.api_url = api_url
        self.api_key = api_key
        self.img_token = img_token
        self.response_times = []

    def generate_image_data(self, image_path):
        # Given an image_path, return image data
        if str(image_path).lower().endswith("tif"):
            with Image.open(image_path) as img:
                img_byte_arr = io.BytesIO()
                img.convert("RGB").save(img_byte_arr, format='JPEG')
                return img_byte_arr.getvalue()
        else:
            with open(image_path, "rb") as image_file:
                return image_file.read()

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        max_tokens=1111,
        count_time=False,
        content_only=True,
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        #print("image_paths", image_paths)
        prompt_parts = prompt.split(self.img_token)
        print("prompt_parts", len(prompt_parts))
        print("image_paths", len(image_paths))
        assert len(prompt_parts) == len(image_paths) + 1

        # Combine all prompt parts into a single question
        question = "".join(prompt_parts)
        #print("question", question)
        if not real_call:
            return {"question": question, "image_paths": image_paths}

        files = [('files', self.generate_image_data(path)) for path in image_paths]
        data = {
            'question': question,
            'api_key': self.api_key
        }

        start_time = time.time()
        
        try:
            response = requests.post(self.api_url, files=files, data=data)
            response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}")
            return None

        end_time = time.time()
        self.response_times.append(end_time - start_time)
        #print("response", response.json())
        if response.status_code == 200:
            result = response.json().get("response", "No response key found in the JSON.")
            print("Response:", result)
            if content_only:
                return result
            else:
                return {
                    "prompt": prompt,
                    "image_paths": image_paths,
                    "response": result,
                    "response_time": end_time - start_time
                }
        else:
            print("Error:", response.status_code, response.text)
            return None
class QWENAPI:
    def __init__(
        self,
        model="qwen-vl-max-0809",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",modal="V",
    ):
        """
        Class for API calls to GPT-series models

        model[str]: the specific model checkpoint to use e.g. "gpt-4-turbo-2024-04-09"
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.modal=modal
        self.client = client = OpenAI(
        api_key="sk-199557b419ab4375985a28a8763fd175",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
        self.token_usage = (0, 0, 0)
        self.response_times = []

    def generate_image_url(self, image_path, detail="low"):
        def encode_image(image_path):
            try:
                with Image.open(image_path) as img:
                    # 统一转换为RGB模式
                    img = img.convert("RGB")
                    # 使用内存中的字节流而不是临时文件
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    return base64.b64encode(buffer.getvalue()).decode("utf-8")
            except Exception as e:
                print(f"Error encoding image {image_path}: {str(e)}")
                return None

        encoded_image = encode_image(image_path)
        if encoded_image is None:
            raise ValueError(f"Failed to encode image: {image_path}")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image}",
                "detail": detail,
            },
        }

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if self.modal=="V" or self.modal=="LV":
            if not isinstance(image_paths, list):  # For single file
                image_paths = [image_paths]
            prompt = prompt.split(self.img_token)
            assert len(prompt) == len(image_paths) + 1
            if prompt[0] != "":
                messages = [self.generate_text_url(prompt[0])]
            else:
                messages = []
            for idx in range(1, len(prompt)):
                messages.append(
                    self.generate_image_url(image_paths[idx - 1], detail=self.detail)
                )
                if prompt[idx].strip() != "":
                    messages.append(self.generate_text_url(prompt[idx]))
            if not real_call:
                return messages
        else:
            #只有语言模态，不需要图片
            # insert code here 
            messages = [self.generate_text_url(prompt)]
            if not real_call:
                return messages
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            #max_tokens=min(4096, max_tokens),
            max_tokens=4096,
            temperature=self.temperature,
            seed=self.seed,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time - start_time]

        self.token_usage = (
            self.token_usage[0] + response.usage.completion_tokens,
            self.token_usage[1] + response.usage.prompt_tokens,
            self.token_usage[2] + response.usage.total_tokens,
        )

        if content_only:
            return response.choices[0].message.content
        else:
            return response
class GPT4VAPI:
    def __init__(
        self,
        model="gpt-4o-mini-2024-07-18",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",modal="V",
    ):
        """
        Class for API calls to GPT-series models

        model[str]: the specific model checkpoint to use e.g. "gpt-4-turbo-2024-04-09"
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.client = OpenAI(api_key=openai_api_key)
        self.token_usage = (0, 0, 0)
        self.response_times = []
        self.modal=modal

    def generate_image_url(self, image_path, detail="low"):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64, {encode_image(image_path)}",
                "detail": detail,
            },
        }

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if self.modal=="V" or self.modal=="LV":
            if not isinstance(image_paths, list):  # For single file
                image_paths = [image_paths]
            prompt = prompt.split(self.img_token)
            assert len(prompt) == len(image_paths) + 1
            if prompt[0] != "":
                messages = [self.generate_text_url(prompt[0])]
            else:
                messages = []
            for idx in range(1, len(prompt)):
                messages.append(
                    self.generate_image_url(image_paths[idx - 1], detail=self.detail)
                )
                if prompt[idx].strip() != "":
                    messages.append(self.generate_text_url(prompt[idx]))
            if not real_call:
                return messages
        else:
            #只有语言模态，不需要图片
            # insert code here 
            messages = [self.generate_text_url(prompt)]
            if not real_call:
                return messages
        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            #max_tokens=min(4096, max_tokens),
            max_tokens=4096,
            temperature=self.temperature,
            seed=self.seed,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time - start_time]

        self.token_usage = (
            self.token_usage[0] + response.usage.completion_tokens,
            self.token_usage[1] + response.usage.prompt_tokens,
            self.token_usage[2] + response.usage.total_tokens,
        )

        if content_only:
            return response.choices[0].message.content
        else:
            return response

class ClaudeAPI:
    def __init__(
        self,
        model="claude-3-5-sonnet-20240620",
        img_token="<<IMG>>",
        seed=66,
        temperature=0,
        detail="auto",
    ):
        """
        Class for API calls to GPT-series models

        model[str]: the specific model checkpoint to use e.g. "gpt-4-turbo-2024-04-09"
        img_token[str]: string to be replaced with images
        seed[int]: seed for generation
        temperature[int]: temperature for generation
        detail[str]: resolution for images. Should be in ['low', 'high', 'auto'].
        """

        self.model = model
        self.img_token = img_token
        self.seed = seed
        self.temperature = temperature
        self.detail = detail
        self.client = anthropic.Anthropic(api_key=claude_api_key)
        self.token_usage = (0, 0, 0)
        self.response_times = []

    def generate_image_url(self, image_path, detail="low"):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
            if str(image_path).lower().endswith("tif"):
                with Image.open(image_path) as img:
                    img.convert("RGB").save("temp.jpeg", "JPEG")
                image_path = "temp.jpeg"
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
            # base64.b64encode(httpx.get(image1_url).content).decode("utf-8")
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type":"image/png",
                "data": encode_image(image_path)
                },
                
            }
        

    def generate_text_url(self, text):
        return {"type": "text", "text": text}

    def __call__(
        self,
        prompt,
        image_paths=[],
        real_call=True,
        count_time=False,
        max_tokens=50,
        content_only=True,
    ):
        """
        Call the API to get the response for given prompt and images
        """
        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert len(prompt) == len(image_paths) + 1
        if prompt[0] != "":
            messages = [self.generate_text_url(prompt[0])]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            messages.append(
                self.generate_image_url(image_paths[idx - 1], detail=self.detail)
            )
            if prompt[idx].strip() != "":
                messages.append(self.generate_text_url(prompt[idx]))
        if not real_call:
            return messages
        start_time = time.time()
        response =self.client.messages.create(
            model=self.model,
            messages=[{"role": "user", "content": messages}],
            max_tokens=min(8192, max_tokens),
            temperature=self.temperature,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, response, end_time - start_time]


        self.token_usage =(response.usage.output_tokens,
                response.usage.input_tokens,
                response.usage.input_tokens + response.usage.output_tokens)

        if content_only:
            return " ".join(block.text for block in response.content if hasattr(block, 'text'))
        else:
            return response
class GeminiAPI:
    def __init__(
        self,
        model="gemini-1.5-pro-preview-0409",
        img_token="<<IMG>>",
        RPM=5,
        temperature=0,
        location="us-central1",
    ):
        """
        Class for API calls to Gemini-series models

        model[str]: the specific model checkpoint to use e.g. "gemini-1.5-pro-preview-0409"
        img_token[str]: string to be replaced with images
        RPM[int]: quota for maximum number of requests per minute
        temperature[int]: temperature for generation
        location[str]: Vertex AI location e.g. "us-central1","us-west1"
        """
        #model="claude-3-5-sonnet@20240620"
        self.model = model
        self.img_token = img_token
        self.temperature = temperature
        vertexai.init(project=gcp_project_id, location=location)
        self.client = GenerativeModel(model)

        self.safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        self.token_usage = (0, 0, 0)

        self.response_times = []
        self.last_time = None
        self.interval = 0.5 + 60 / RPM

    def generate_image_url(self, image_path):
        # Given an image_path, return a dict
        # Function to encode the image
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")

        image1 = Part.from_data(mime_type="image/jpeg", data=encode_image(image_path))
        return image1

    def __call__(
        self, prompt, image_paths=[], real_call=True, max_tokens=50, content_only=True
    ):
        """
        Call the API to get the response for given prompt and images
        """

        if self.last_time is not None:  # Enforce RPM
            # Calculate how much time the loop took
            end_time = time.time()
            elapsed_time = end_time - self.last_time
            # Wait for the remainder of the interval, if necessary
            if elapsed_time < self.interval:
                time.sleep(self.interval - elapsed_time)

        if not isinstance(image_paths, list):  # For single file
            image_paths = [image_paths]
        prompt = prompt.split(self.img_token)
        assert len(prompt) == len(image_paths) + 1
        if prompt[0] != "":
            messages = [prompt[0]]
        else:
            messages = []
        for idx in range(1, len(prompt)):
            messages.append(self.generate_image_url(image_paths[idx - 1]))
            if prompt[idx].strip() != "":
                messages.append(prompt[idx])
        if not real_call:
            return messages

        start_time = time.time()
        self.last_time = start_time
        responses = self.client.generate_content(
            messages,
            generation_config={
                "max_output_tokens": min(max_tokens, 8192),
                "temperature": self.temperature,
            },
            safety_settings=self.safety_settings,
            stream=False,
        )

        end_time = time.time()
        self.response_times.append(end_time - start_time)

        results = [prompt, image_paths, responses, end_time - start_time]

        try:
            usage = responses._raw_response.usage_metadata
            self.token_usage = (
                self.token_usage[0] + usage.candidates_token_count,
                self.token_usage[1] + usage.prompt_token_count,
                self.token_usage[2] + usage.total_token_count,
            )
        except:
            pass
        if content_only:
            return responses.text
        else:
            return responses

     