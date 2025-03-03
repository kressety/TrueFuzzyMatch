import numpy as np
import requests

# 配置 Ollama 的 API 地址和模型
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "modelscope.cn/ZackZHU/gte-Qwen2-7B-instruct-GGUF-f16:latest"


# 检查 Ollama 连接
def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json()["models"]
            model_names = [m["name"] for m in models]
            print("Ollama 服务正常运行，可用模型:", model_names)
            if MODEL_NAME in model_names:
                print(f"模型 {MODEL_NAME} 已加载，可以使用。")
            else:
                raise Exception(f"模型 {MODEL_NAME} 未找到，请检查。")
        else:
            raise Exception(f"Ollama 服务连接失败，状态码: {response.status_code}")
    except requests.ConnectionError:
        raise Exception("无法连接到 Ollama，请确保服务已启动。")


# 计算嵌入向量
def get_embedding(text, model=MODEL_NAME):
    payload = {"model": model, "prompt": text}
    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return np.array(response.json()["embedding"])
    else:
        raise Exception(f"Failed to get embedding: {response.text}")
