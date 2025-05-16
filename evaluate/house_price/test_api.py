import os
import sys
import httpx
from openai import OpenAI
import openai
from openai import AzureOpenAI
PROXY = "http://127.0.0.1:10190"
DEEPINFRA_APIKEY = os.environ["DeepInfra_API_KEY"]
SILICONFLOW_APIKEY = os.environ["SiliconFlow_API_KEY"]
client = AzureOpenAI(
            api_key = "AX4weTYHkKjMP9s2pmuIicmjxjoY0Oek96n9ILMSD3a9rzXCSjlEJQQJ99BBACHYHv6XJ3w3AAABACOGQaCI",  # api key  
            api_version = "2025-01-01-preview",
            azure_endpoint = "https://liutianhui-citygpt-gpt4omini.openai.azure.com/"  # end point
        )
# 
content = []
    
    # 先添加文本部分
# content.append({
#     "type": "text",
#     "text": "Suppose you are a professional crime data analyst in newyork, United States. Based on the provided satellite imagery and several street view photos, please estimate the occurrence of violent crime in the census tract where these images are taken. Consider factors such as location, visible property features, neighborhood condition, and any other relevant details that might influence the occurrence of violent crime in the area."
# })
# for i in range(1, 20):
# content.append({
#     "type": "text",
#     "text": "Hello, what's your name?"
# })
content.append({
      "type": "image_url",
      "image_url": {
          "url": "http://streetv.oss-ap-northeast-1.aliyuncs.com/US%2F0_--Sdgpc7RwfjWHI1_8JKoQ%2633.35721787265349%26-111.7601800341597%2615.jpg?OSSAccessKeyId=LTAI5t9ccSKeSPyP598pEH5B&Expires=1748976438&Signature=zj%2FELSyyN9zpitsgg7IlbpDFUy8%3D"
      },
      "detail": "high"
  })

session = [{
        "role": "user",
        "content": content
    }]
# response = client.chat.completions.create(
#   model="o4-mini",
#   messages=[
#     {"role": "user", "content": content}
#   ]
# )    
# client = OpenAI(
#         base_url="https://api.deepinfra.com/v1/openai",
#         api_key=DEEPINFRA_APIKEY,
#         http_client=httpx.Client(proxies=PROXY),
#             )
# client = OpenAI(
#         api_key=SILICONFLOW_APIKEY,
#         base_url="https://api.siliconflow.cn/v1"
#         )
# google/gemini-2.5-flash google/gemini-2.5-pro
model_name = "gpt-4.1-nano"
response = client.chat.completions.create(
                        model=model_name,
                        messages=session,
                        temperature=0,
                        max_tokens=200,
                    )
answer = response.choices[0].message.content
print(answer)
if hasattr(response, 'usage'):
    usage = response.usage
    print("\n🔸 Token 使用统计：")
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
else:
    print("⚠️ 无法获取 usage 信息，可能是模型版本不支持或 API 响应格式变化")
sys.exit(0)
