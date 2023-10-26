from typing import Iterator
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import asyncio
import os
from datetime import datetime
import pandas as pd
import torch
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain.text_splitter import (
    MarkdownTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.vectorstores.chroma import Chroma
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)


# 刘平学长给定的分割方式
def _initDocumentManager(name: str):
    rank = {
        "报销管理制度-202112.docx": 0.6,
        "便民问答整理_waic.docx": 1,
        "补充问题及答案_waic.docx": 1,
        "采购管理办法202112.docx": 0.6,
        "叠境财务制度-202112.docx": 0.6,
        "叠境数字科技（上海）有限公司介绍_新闻稿.docx": 1,
        "叠境数字员工手册_waic.docx": 0.8,
        "额外信息_waic.docx": 1,
        "固定资产管理办法202112.docx": 0.6,
        "问答题库全集_LpClear.docx": 1,
        "虞老师公关_标准问答_0617.docx": 1,
        "2.医保-便民问答整理(2).doc": 0.6,
        "叠境数字员工手册(分隔后）_waic.docx": 0.8,
    }
    # rank = RANK

    try:
        res = rank[name]
        return res
    except:
        print(f"{name} document is not in rank board.")
        return 0.6


def _buildChromaVectorDB(
    embeddings, vector_db_path, file_list: List[str], file_folder, qwenModel
):
    ##TODO:使用qwen缩写之后构建向量库
    data = None
    data_rank1 = None
    data_others = None
    text_splitter = None
    for file in file_list:
        file_data = None
        file_extension = os.path.splitext(file)[1]#get types of files
        file_name = os.path.basename(file)

        if file_extension == ".txt":
            loader = TextLoader(file)
            file_data = loader.load()
        elif file_extension == ".md":
            loader = UnstructuredMarkdownLoader(file)
            file_data = loader.load()
            text_splitter = MarkdownTextSplitter()
        elif file_extension == ".csv":
            loader = CSVLoader(file)
            file_data = loader.load()
        elif file_extension == ".pdf":
            loader = PyPDFLoader(file)
            file_data = loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file)
            ##TODO:file_data???
            file_data = loader.load()
        elif file_extension == ".xlsx":
            continue

        if _initDocumentManager(file_name) == 2:
            if data_rank1 == None:
                data_rank1 = file_data
            else:
                for page in file_data:
                    data_rank1.append(page)
        else:
            if data_others == None:
                data_others = file_data
            else:
                ##TODO:page???
                for page in file_data:
                    data_others.append(page)

    if text_splitter is None:
        text_splitter_rank1 = CharacterTextSplitter(
            separator="~DgeneDgene~", chunk_size=500, chunk_overlap=0
        )
        text_splitter_others = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=20
        )

    splits_rank1 = []
    if data_rank1 is not None:
        splits_rank1 = text_splitter_rank1.split_documents(data_rank1)
    if data_others is not None:
        ##TODO:how to split
        splits_others = text_splitter_others.split_documents(data_others)
        for page in splits_others:
            splits_rank1.append(page)
    splits = splits_rank1
    xlsx_file_path = os.path.join(file_folder, "问答题库全集_LpClear.xlsx")
    data_frame = pd.read_excel(xlsx_file_path, header=None)

    # 遍历每一行并进行处理
    for _, row in data_frame.iterrows():
        splits.append(Page_(f"Q:{row[0]}\nA:{row[1]}", {"source": "问答全集.xlsx"}))

    # 构建向量库
    docsearch = Chroma.from_documents(
        splits, embeddings, persist_directory=vector_db_path
    )
    return docsearch


class Page_(dict):
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredImageLoader,
    UnstructuredHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
)
from langchain.text_splitter import (
    MarkdownTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
import os
from langchain.embeddings.huggingface import (
    HuggingFaceInstructEmbeddings,
    HuggingFaceEmbeddings,
)
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.generation import GenerationConfig


def load_qwen(qwen_path):
    print("Loading Tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    print("Completed ...")
    print("Loading qwen-7b-chat Model ...")
    # 打开bf16精度，A100、H100、RTX3060、RTX3070等显卡建议启用以节省显存
    model = AutoModelForCausalLM.from_pretrained(
        qwen_path,
        device_map="auto",
        trust_remote_code=True,  # bf16=True
    ).eval()
    # 打开fp16精度，V100、P100、T4等显卡建议启用以节省显存
    # model = AutoModelForCausalLM.from_pretrained("../../Qwen-7B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # 使用CPU进行推理，需要约32GB内存
    # model = AutoModelForCausalLM.from_pretrained("../../Qwen-7B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # 默认使用自动模式，根据设备自动选择精度
    # # 可指定不同的生成长度、top_p等相关超参
    model.generation_config = GenerationConfig.from_pretrained(
        qwen_path, trust_remote_code=True
    )
    model.generation_config.temperature = 0.8

    print("Qwen Loaded Completed")
    return tokenizer, model


def load_bge(model_path, device):
    print("start to load bge-large-zh")
    # 加载模型
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path, model_kwargs={"device": device}
    )
    print("bge-large-zh loaded")
    return embeddings


def load_vectorstore(embeddings, vector_db_path, file_folder, qwenModel):
    if os.path.exists(vector_db_path):  # 如果路径存在向量库，则加载
        print(
            f"The vector database has already been established at {vector_db_path}, load the vector repository"
        )
        return Chroma(persist_directory=vector_db_path, embedding_function=embeddings)
    else:
        print(
            f"The input VectorStore is not found, Start to build at {vector_db_path} "
        )  # 如果向量库不存在，则构建
        list_files = os.listdir(file_folder)
        list_files = [os.path.join(file_folder, file) for file in list_files]
        return _buildChromaVectorDB(
            embeddings, vector_db_path, list_files, file_folder, qwenModel
        )


class Qwen:
    PROMPT_TEMPLATE = """你是一名专业的HR，请全面地参考给定的问题，在不丢失文本含义情况下，用中文简要概括回答下面的问题：
    用户: {question}
    你的概括回答："""

    def __init__(
        self,
        qwen_path,
    ) -> None:
        # import torch

        # __import__("pysqlite3")
        # import sys

        # sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
        # self.log_path = log_path
        # torch.cuda.empty_cache()
        # 确定model以及向量库路径                                                             # model名称
        # 向量库路径

        # 知识库路径

        # 加载embedding模型
        # EMBEDDING_DEVICE = (
        #     "cuda"
        #     if torch.cuda.is_available()
        #     else "mps"
        #     if torch.backends.mps.is_available()
        #     else "cpu"
        # )
        # EMBEDDING_DEVICE='cpu'
        # embeddings = load_bge(bge_path, EMBEDDING_DEVICE)

        # 加载qwen7bchat模型
        self._tokenizer, self._model = load_qwen(qwen_path)

        # # 加载\构造本地向量库
        # self._db = load_vectorstore(
        #     embeddings=embeddings,
        #     vector_db_path=vector_db_path,
        #     file_folder=file_folder,
        # )

    # def _get_ref_content(self, question):
    #     # 基于向量库搜索相关参考，构造prompt并返回
    #     # str -> str
    #     refs = self._db.similarity_search(question)
    #     temp = "\n################\n".join([doc.page_content for doc in refs][::-1])
    #     return temp

    # def _get_all_query(self, question, contexts):
    #     # 构造prompt
    #     # str -> str
    #     return self.PROMPT_TEMPLATE.replace("{contexts}", contexts).replace(
    #         "{question}", question
    #     )

    def _get_all_query(self, question):
        # 构造prompt
        # str -> str
        return self.PROMPT_TEMPLATE.replace("{question}", question)

    def chat(self, query: str):
        dir = os.path.join(self.log_path, self.__class__.__name__)
        if not os.path.exists(dir):
            os.makedirs(dir)
        time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3]
        filepath = os.path.join(dir, str(time) + ".txt")
        with open(filepath, "w") as file:
            file.write(f"Time: {time}\n")
            file.write("********************\n")
            file.write(f"Source Document: {query}\n")
            # refs = self._get_ref_content(query)
            query = self._get_all_query(query)  # 构造prompt问题
            # 流式输出，目前还没测试history功能
            file.write("********************\n")
            file.write(f"{query}\n")
            file.write("********************\n")
            response, history = self._model.chat(self._tokenizer, query, history=None)
            file.write(f"Summary Document: {response}\n")
        return response


async def _main(
    QWEN_PATH,
    VECTOR_DB_PATH,
    FILE_FOLDER,
    BGE_PATH,
    ws_logger_dir,
):
    import logging

    if not os.path.exists(ws_logger_dir):
        os.makedirs(ws_logger_dir)
    logfile = "ws.log"
    from functools import partial

    qwen_factory = partial(
        Qwen,
        qwen_path=QWEN_PATH,
        vector_db_path=VECTOR_DB_PATH,
        file_folder=FILE_FOLDER,
        bge_path=BGE_PATH,
        log_path=ws_logger_dir,
    )
    ws_logger = logging.getLogger("llmws_dgene")
    ws_logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(
        filename=os.path.join(ws_logger_dir, logfile), encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ws_logger.addHandler(fh)
    console_handler = logging.StreamHandler()
    ws_logger.addHandler(console_handler)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--QWEN_PATH", type=str)
    parser.add_argument("--VECTOR_DB_PATH", type=str)
    parser.add_argument("--FILE_FOLDER", type=str)
    parser.add_argument("--BGE_PATH", type=str)
    args = parser.parse_args()

    qwen_path = args.QWEN_PATH
    bge_path = args.BGE_PATH
    vector_db_path = args.VECTOR_DB_PATH
    file_folder = args.FILE_FOLDER

    qwen = Qwen(qwen_path=qwen_path)
    # 加载embedding模型
    EMBEDDING_DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    embeddings = load_bge(bge_path, EMBEDDING_DEVICE)

    # 加载\构造本地向量库
    db = load_vectorstore(
        embeddings=embeddings,
        vector_db_path=vector_db_path,
        file_folder=file_folder,
        qwenModel=qwen,
    )
