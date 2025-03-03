from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from ollama_config import check_ollama_connection, get_embedding

A_table_path = "6_精简后的物料库（自采H物料）（2404行+工程773行+厨房查缺补漏884行）.xlsx"
A_sheet_name = "物料编码表"


# 并行计算嵌入
def compute_embeddings(texts, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(tqdm(executor.map(get_embedding, texts), total=len(texts)))
    return embeddings


# 主程序：生成 A 表嵌入
if __name__ == "__main__":
    print("检查 Ollama 连接...")
    check_ollama_connection()

    A_df = pd.read_excel(A_table_path, sheet_name=A_sheet_name)
    print("A 表列名:", A_df.columns.tolist())
    print(f"A 表行数: {A_df.shape[0]}")

    # 计算 A 表 '物料名称' 列的嵌入（根据你的任务描述，应为 B 列对应 '物料名称'）
    print("正在计算 A 表 '物料名称' 列的嵌入向量（多线程）...")
    A_texts = [str(text) for text in A_df['物料名称'].fillna("")]
    A_embeddings = compute_embeddings(A_texts, max_workers=8)

    # 校验嵌入数量
    if len(A_embeddings) != A_df.shape[0]:
        raise ValueError(f"嵌入数量 ({len(A_embeddings)}) 与 A 表行数 ({A_df.shape[0]}) 不匹配！")

    # 保存嵌入到文件（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_output_path = f"A_table_embeddings_{timestamp}.npy"
    np.save(embedding_output_path, A_embeddings)
    print(f"A 表嵌入已保存到 {embedding_output_path}")
