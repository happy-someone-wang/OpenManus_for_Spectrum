# preprocessor.py

import os
import re
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# ----------------------
# 配置导入（建议统一配置）
# ----------------------
from config import knowledge_base_path, local_embedding_path, device,vector_db_path


class SmartDocumentProcessor:
    def __init__(self):
        # 初始化嵌入模型
        self.model = HuggingFaceEmbeddings(
            model_name=local_embedding_path,
            model_kwargs={
                'device': device,
                'trust_remote_code': True
            },
            encode_kwargs={
                'normalize_embeddings': True,
                'query_instruction': "为这个句子生成表示以用于检索相关文章："
            }
        )

    def _detect_content_type(self, text):
        if re.search(r'def |import |print\(|代码示例', text):
            return "code"
        elif re.search(r'\|.+\|', text) and '%' in text:
            return "table"
        return "normal"

    def process_documents(self):
        # 加载文档
        loaders = [
            DirectoryLoader(knowledge_base_path, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(knowledge_base_path, glob="**/*.txt", loader_cls=TextLoader)
        ]
        documents = []
        for loader in loaders:
            documents.extend(loader.load())

        # 语义分块
        chunker = SemanticChunker(
            embeddings=self.model,
            breakpoint_threshold_amount=82,
            add_start_index=True
        )
        base_chunks = chunker.split_documents(documents)

        # 动态二次分块
        final_chunks = []
        for chunk in base_chunks:
            content_type = self._detect_content_type(chunk.page_content)
            if content_type == "code":
                splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=64)
            elif content_type == "table":
                splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=96)
            else:
                splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            final_chunks.extend(splitter.split_documents([chunk]))

        # 添加元数据
        for i, chunk in enumerate(final_chunks):
            chunk.metadata.update({
                "chunk_id": f"chunk_{i}",
                "content_type": self._detect_content_type(chunk.page_content)
            })

        return final_chunks

    def create_vector_db(self, chunks, persist_directory = vector_db_path):
        # 创建向量数据库
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.model,
            persist_directory=persist_directory
        )
        return vector_db

if __name__ == "__main__":
    processor = SmartDocumentProcessor()
    chunks = processor.process_documents()
    processor.create_vector_db(chunks)
