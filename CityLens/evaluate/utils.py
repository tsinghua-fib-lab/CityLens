import json
from config import SILICONFLOW_APIKEY, DEEPINFRA_APIKEY, PROXY, PROXIES
import time
import httpx
import requests
from openai import OpenAI
from openai import AzureOpenAI
openai_api_key = ""
openai_end_point = ""
openroute_api_key =""

def get_response_mllm_api(session, model_name, temperature=0, max_tokens=1000, infer_server=None,json_mode=False):
    max_retries = 5
    retries = 0
    if model_name in ["deepseek-ai/deepseek-vl2", "Qwen/Qwen2.5-VL-32B-Instruct", "Qwen/Qwen2.5-VL-72B-Instruct", "Pro/Qwen/Qwen2.5-VL-7B-Instruct"]:
        infer_server = "Siliconflow"
        # print(infer_server)
        client = OpenAI(
        api_key=SILICONFLOW_APIKEY,
        base_url="https://api.siliconflow.cn/v1"
        )
    elif model_name in ["Qwen2.5-VL-7B"]:
        infer_server = "local"
        # print(infer_server)
        client = OpenAI(
        api_key="token-fiblab-20250508",
        base_url="http://106.63.100.85:18259/v1"
        )
    elif model_name in ["Qwen2.5-VL-3B"]:
        infer_server = "local"
        # print(infer_server)
        client = OpenAI(
        api_key="token-fiblab-20250508",
        base_url="http://106.63.100.85:18258/v1"
        )
    elif model_name in ["meta-llama/Llama-3.2-90B-Vision-Instruct", "meta-llama/Llama-3.2-11B-Vision-Instruct", "google/gemma-3-27b-it", "google/gemma-3-12b-it", "google/gemma-3-4b-it", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "google/gemini-1.5-flash", "google/gemini-2.5-flash", "google/gemini-2.5-pro", "google/gemini-2.0-flash-001"]:
        infer_server = "DeepInfra"
        client = OpenAI(
        base_url="https://api.deepinfra.com/v1/openai",
        api_key=DEEPINFRA_APIKEY,
        http_client=httpx.Client(proxies=PROXY),
            )
    elif model_name in ["gpt-4o", "gpt-4.1", "gpt-4.1-mini", "gpt-4o-mini", "o4-mini", "gpt-4.1-nano"]:
        infer_server = "OpenAI"
        client = AzureOpenAI(
            api_key = openai_api_key,  # api key  
            api_version = "2025-01-01-preview",
            azure_endpoint = openai_end_point  # end point
        )
    elif model_name in ["qwen/qwen2.5-vl-72b-instruct", "qwen/qwen2.5-vl-32b-instruct", "qwen/qwen-2.5-vl-7b-instruct", "openai/gpt-4.1-mini", "openai/gpt-4.1-nano", "google/gemini-2.5-flash-preview", "opengvlab/internvl3-14b:free", "qwen/qwen-vl-plus", "mistralai/mistral-small-3.1-24b-instruct", "microsoft/phi-4-multimodal-instruct", "mistralai/pixtral-12b", "amazon/nova-lite-v1"]:
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openroute_api_key}",
            "Content-Type": "application/json"
        }
        # print(model_name)
        payload = {
            "model": model_name,
            "messages": session,
            "provider": {
                'sort': 'price',
                "ignore": ["Together", "Cloudflare"]
            },
        }
        while retries < max_retries:
            try:
                # print("here")
                response = requests.post(url, headers=headers, json=payload, proxies=PROXIES)
                # print("response: ", response)
                print("response: ", response.json())
                content = response.json()["choices"][0]["message"]["content"]
                # content = response.text
                # 
                # print(content)
                return content
            except Exception as e:
                retries += 1
                print(f"Error calling GPT API: {e}. Retry {retries}/{max_retries}...")

                if retries < max_retries:
                    time.sleep(2 ** retries)  # 指数回退，等待一段时间再重试
        return None

    
    while retries < max_retries:
        try:
            if json_mode:
                response = client.chat.completions.create(
                            model=model_name,
                            response_format={"type": "json_object"},
                            messages=session,
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                client.close()
                return response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                        model=model_name,
                        messages=session,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                client.close()
                # print(response.usage.prompt_tokens, response.usage.completion_tokens)
                # print(response.choices[0].message.content)
                return response.choices[0].message.content
        except Exception as e:
            retries += 1
            print(f"Error calling GPT API: {e}. Retry {retries}/{max_retries}...")

            if retries < max_retries:
                time.sleep(2 ** retries)  # 指数回退，等待一段时间再重试
    client.close()
    return None
from PIL import Image
import io
import base64
def convert_image_to_webp_base64(input_image_path):
    try:
        with Image.open(input_image_path) as img:
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='webp')
            byte_arr = byte_arr.getvalue()
            base64_str = base64.b64encode(byte_arr).decode('utf-8')
            return base64_str
    except IOError:
        print(f"Error: Unable to open or convert the image {input_image_path}")
        return None
