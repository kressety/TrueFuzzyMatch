import numpy as np
import pandas as pd
from tqdm import tqdm

from ollama_config import check_ollama_connection, get_embedding

A_table_path = "6_精简后的物料库（自采H物料）（2404行+工程773行+厨房查缺补漏884行）.xlsx"
A_sheet_name = "物料编码表"
B_table_path = "检索表.xlsx"
B_sheet_name = "物料列表"
embedding_input_path = "A_table_embeddings_20250303_140802.npy"


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# 主程序：检索 B 表
if __name__ == "__main__":
    print("检查 Ollama 连接...")
    check_ollama_connection()

    A_df = pd.read_excel(A_table_path, sheet_name=A_sheet_name)
    A_embeddings = np.load(embedding_input_path, allow_pickle=True)
    print("A 表列名:", A_df.columns.tolist())
    print(f"已加载 A 表嵌入，数量: {len(A_embeddings)}")
    if len(A_embeddings) != A_df.shape[0]:
        raise ValueError(f"嵌入数量 ({len(A_embeddings)}) 与 A 表行数 ({A_df.shape[0]}) 不匹配！")

    if not B_table_path:
        raise ValueError("请在 B_table_path 中指定 B 表文件名！")
    B_df = pd.read_excel(B_table_path, sheet_name=B_sheet_name)  # 填写 B 表 sheet 名
    print("B 表列名:", B_df.columns.tolist())

    # 逐行处理 B 表
    print("开始处理 B 表...")
    for index, row in tqdm(B_df.iterrows(), total=B_df.shape[0]):
        query_text = str(row['物料名称'])  # B 表 A 列，可能需要改为实际列名
        query_embedding = get_embedding(query_text)
        similarities = [cosine_similarity(query_embedding, emb) for emb in A_embeddings]
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]  # 获取最高相似度值
        best_match_row = A_df.iloc[best_match_idx]
        B_df.at[index, '物料编码'] = best_match_row['物料编码']  # A 表的 A 列（物料编码）
        B_df.at[index, '计量单位编码'] = best_match_row['计量单位编码']  # A 表的 C 列（计量单位编码）
        B_df.at[index, '匹配物料名称'] = best_match_row['物料名称']  # A 表的 B 列（物料名称）
        B_df.at[index, '相似度'] = best_similarity  # 新增：将相似度写入B表

    # 保存结果
    output_path = f"O-{B_table_path}"
    B_df.to_excel(output_path, index=False)
    print(f"处理完成，结果已保存到 {output_path}")
