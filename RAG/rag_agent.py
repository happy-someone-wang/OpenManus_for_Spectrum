from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import os
os.environ['CUDA_VISIBLE_DEVICE']='1'
# ----------------------
# 配置导入
# ----------------------
from config import LOCAL_EMBEDDING_PATH, LOCAL_RARANKER_PATH, LOCAL_LLM_PATH,VECTOR_DB_PATH, DEVICE

class HybridRetriever:
    def __init__(self, persist_directory = VECTOR_DB_PATH):
        # 加载向量数据库
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(model_name=LOCAL_EMBEDDING_PATH)
        )

        # 创建BM25检索器，从文档块中初始化，初始检索数量为5
        docs = self.vector_db.get(include=["documents", "metadatas"])
        texts = docs["documents"]
        metadatas = docs["metadatas"]
        self.bm25_retriever = BM25Retriever.from_texts(
            texts=texts,
            metadatas=metadatas,
            k=5
        )

        # 创建混合检索器，结合向量和BM25检索，权重分别为0.6和0.4
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[
                self.vector_db.as_retriever(search_kwargs={"k": 5}),
                self.bm25_retriever
            ],
            weights=[0.6, 0.4]
        )

        # 初始化重排序模型，使用BAAI/bge-reranker-large
        self.reranker = CrossEncoder(
            model_name_or_path = LOCAL_RARANKER_PATH,  # 本地模型路径
            device= DEVICE if torch.cuda.is_available() else "cpu"  # 自动设备选择
        )

    def retrieve(self, query, top_k=3):
        # 第一阶段：使用混合检索器获取相关文档
        docs = self.ensemble_retriever.get_relevant_documents(query)
        # docs = self.ensemble_retriever.invoke(query)

        # 第二阶段：为查询和每个文档创建配对，用于重排序
        pairs = [[query, doc.page_content] for doc in docs]
        scores = self.reranker.predict(pairs)
        # 使用重排序模型预测配对的分数
        ranked_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

        # 返回top_k结果
        return [doc for doc, _ in ranked_docs[:top_k]]

# ----------------------
# 3. RAG系统集成
# ----------------------
class RAGConfig:
    model_name = LOCAL_LLM_PATH  # 默认本地模型
    device = DEVICE
    max_length = 4096
    temperature = 0.1
    top_p = 0.9

class ModelLoader:
    def __init__(self, config):
        self.config = config
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE,
            trust_remote_code=True,
            local_files_only=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        return model, tokenizer

    def generate_local(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.config.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(outputs  [0], skip_special_tokens=True)

class EnhancedRAG:
    def __init__(self,config):
        # 初始化检索器
        self.retriever = HybridRetriever()

        # 初始化模型加载器
        self.config = config
        self.model_loader = ModelLoader(config)

        # 如果使用本地模型，初始化pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model_loader.model,
            tokenizer=self.model_loader.tokenizer,
            torch_dtype=torch.bfloat16
        )

    def generate_prompt(self, question, contexts):
        # 格式化上下文，包含来源和类型信息
        context_str = "\n\n".join([
            f"[来源：{doc.metadata['source']}，类型：{doc.metadata['content_type']}]\n{doc.page_content}"
            for doc in contexts
        ])
        # 创建提示模板，要求基于上下文回答问题
        return f"""你是一个专业助手，请严格根据以下带来源的上下文：
        {context_str}

        按步骤思考并回答：{question}
        如果上下文信息不足，请明确指出缺失的信息。最后用中文给出结构化答案。"""

    def post_process(self, text):
        # 移除特殊token和多余的空格
        text = text.replace("<|begin_of_text|>", "").replace("<|eot_id|>", "").strip()
        # 按行分割文本
        lines = text.split('\n')

        # 遍历每一行，查找以指定前缀开头的行
        for i, line in enumerate(lines):
            if line.startswith('        如果上下文信息不足，请明确指出缺失的信息。'):
                # 截取该行之后的内容，并拼接成字符串
                return '\n'.join(lines[i+1:])

        # 如果没有找到匹配的行，返回原始文本或空字符串
        return text

    # RAG生成
    def rag_generate(self,question):
        # 使用检索器获取与问题相关的上下文
        contexts = self.retriever.retrieve(question)
        # 根据问题和上下文生成提示
        rag_prompt = self.generate_prompt(question, contexts)
        return rag_prompt

    def ask(self, question):
        # 生成提示
        prompt = self.rag_generate(question)

        # print(f"RAG结果：{prompt}")

        # 本地模型回答
        response = self.model_loader.generate_local(prompt)
        return self.post_process(response)

if __name__ == "__main__":
    config = RAGConfig()
    rag = EnhancedRAG(config)
    complex_question = "我现在正在学习电磁频谱作战的有关概念，请假定一个作战场景，并针对频谱管控的相关问题给出策略"
    answer = rag.ask(complex_question)
    print(f"问题：{complex_question}")
    print("***********************分隔线***********************")
    print("答案：")
    print(answer)
