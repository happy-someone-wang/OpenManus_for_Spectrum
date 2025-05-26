# config.py

import torch

# 设备配置
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# 路径配置
LOCAL_EMBEDDING_PATH = "/home/wangyangle/workspace2/wangyangle/ragllm/OpenManus_for_Spectrum/RAG/models/bge-small-zh-v1.5"  # 本地嵌入模型路径
LOCAL_LLM_PATH = "/home/wangyangle/workspace2/wangyangle/ragllm/OpenManus_for_Spectrum/RAG/models/Qwen3-4B-Base"  # 本地大模型路径
LOCAL_RERANKER_PATH = "/home/wangyangle/workspace2/wangyangle/ragllm/OpenManus_for_Spectrum/RAG/models/bge-reranker-base" #本地重排序模型路径
KNOELEDGE_BASE_PATH = "/home/wangyangle/workspace2/wangyangle/ragllm/OpenManus_for_Spectrum/RAG/knowledge_base"
VECTOR_DB_PATH = "/home/wangyangle/workspace2/wangyangle/ragllm/OpenManus_for_Spectrum/RAG/vector_db"

