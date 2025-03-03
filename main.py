import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from tqdm import tqdm

# 初始化 Rich Console
console = Console()

# 配置 Ollama 的 API 地址
OLLAMA_API_URL = "http://localhost:11434/api/embeddings"
DEFAULT_MAX_WORKERS = 8


# 检查 Ollama 连接并获取模型列表
def check_ollama_connection():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not models:
                console.print("[red]Ollama 服务未返回任何模型！[/red]")
                return []
            return [m["name"] for m in models]
        else:
            console.print(f"[red]Ollama 服务连接失败，状态码: {response.status_code}[/red]")
            return []
    except requests.ConnectionError:
        console.print("[red]无法连接到 Ollama，请确保服务已启动！[/red]")
        return []
    except Exception as e:
        console.print(f"[red]检查 Ollama 连接时发生未知错误: {e}[/red]")
        return []


# 获取嵌入向量（无超时和重试）
def get_embedding(text, model):
    try:
        payload = {"model": model, "prompt": text}
        response = requests.post(OLLAMA_API_URL, json=payload)
        if response.status_code == 200:
            embedding = np.array(response.json()["embedding"])
            if embedding.size == 0 or not isinstance(embedding, np.ndarray):
                raise ValueError("Ollama 返回了无效的嵌入向量")
            return embedding
        else:
            raise Exception(f"获取嵌入失败: {response.text}")
    except Exception as e:
        console.print(f"[yellow]嵌入计算警告: {e} (文本: {text[:50]})[/yellow]")
        return None


# 并行计算嵌入（显示总行数）
def compute_embeddings(texts, model, max_workers=DEFAULT_MAX_WORKERS):
    try:
        embeddings = [None] * len(texts)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_embedding, text, model): i for i, text in enumerate(texts)}
            for future in tqdm(futures, total=len(texts), desc="Computing embeddings"):
                idx = futures[future]
                try:
                    embeddings[idx] = future.result()
                except Exception as e:
                    console.print(f"[yellow]嵌入计算失败 (索引 {idx}): {e}[/yellow]")
                    embeddings[idx] = None
        failed_count = sum(1 for emb in embeddings if emb is None)
        if failed_count > 0:
            console.print(f"[yellow]警告: {failed_count}/{len(texts)} 个嵌入计算失败[/yellow]")
        return embeddings
    except Exception as e:
        console.print(f"[red]嵌入计算严重失败: {e}[/red]")
        return None


# 并行计算余弦相似度
def compute_similarities(query_embedding, embeddings, max_workers=DEFAULT_MAX_WORKERS):
    try:
        if query_embedding is None:
            return [0.0] * len(embeddings)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            similarities = list(
                executor.map(lambda emb: cosine_similarity(query_embedding, emb) if emb is not None else 0.0,
                             embeddings))
        return similarities
    except Exception as e:
        console.print(f"[red]相似度计算失败: {e}[/red]")
        return None


# 计算余弦相似度
def cosine_similarity(vec1, vec2):
    try:
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    except Exception as e:
        console.print(f"[red]余弦相似度计算错误: {e}[/red]")
        return 0.0


# 列出当前目录下的文件
def list_files(extension):
    try:
        files = [f for f in os.listdir() if f.endswith(extension)]
        if not files:
            console.print(f"[yellow]当前目录下没有 {extension} 文件！[/yellow]")
        return files
    except Exception as e:
        console.print(f"[red]读取目录失败: {e}[/red]")
        return []


# 处理 A 表：生成嵌入
def process_A(model):
    console.print("[bold green]任务 A: 生成 A 表嵌入[/bold green]")
    xlsx_files = list_files(".xlsx")
    if not xlsx_files:
        return

    table = Table(title="Available .xlsx Files")
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="magenta")
    for i, file in enumerate(xlsx_files):
        table.add_row(str(i), file)
    console.print(table)

    file_idx = IntPrompt.ask("请选择一个 .xlsx 文件（输入编号）", choices=[str(i) for i in range(len(xlsx_files))],
                             default=0)
    A_table_path = xlsx_files[file_idx]

    try:
        A_df = pd.read_excel(A_table_path)
    except Exception as e:
        console.print(f"[red]加载 {A_table_path} 失败: {e}[/red]")
        return

    console.print(f"已加载文件: {A_table_path}")
    console.print("A Table Columns:", A_df.columns.tolist())

    col_idx = IntPrompt.ask("请选择 '物料名称' 所在的列（输入编号）", choices=[str(i) for i in range(len(A_df.columns))],
                            default=0)
    col_name = A_df.columns[col_idx]

    console.print(f"正在计算 '{col_name}' 列的嵌入向量（多线程）...")
    A_texts = [str(text) for text in A_df[col_name].fillna("")]
    A_embeddings = compute_embeddings(A_texts, model)
    if A_embeddings is None or len(A_embeddings) != A_df.shape[0]:
        console.print("[red]嵌入生成失败或数量不匹配！[/red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedding_output_path = f"A_table_embeddings_{os.path.splitext(A_table_path)[0]}_{timestamp}.npy"
    try:
        np.save(embedding_output_path, A_embeddings)
        console.print(f"[green]A 表嵌入已保存到 {embedding_output_path}[/green]")
    except Exception as e:
        console.print(f"[red]保存嵌入失败: {e}[/red]")


# 处理 B 表：匹配物料
def process_B(model):
    console.print("[bold green]任务 B: 匹配 B 表物料[/bold green]")
    xlsx_files = list_files(".xlsx")
    npy_files = list_files(".npy")
    if not xlsx_files or not npy_files:
        return

    # 选择 A 表
    table = Table(title="Available .xlsx Files (A Table)")
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="magenta")
    for i, file in enumerate(xlsx_files):
        table.add_row(str(i), file)
    console.print(table)
    a_file_idx = IntPrompt.ask("请选择 A 表 .xlsx 文件（输入编号）", choices=[str(i) for i in range(len(xlsx_files))],
                               default=0)
    A_table_path = xlsx_files[a_file_idx]

    # 自动匹配 .npy 文件
    A_base_name = os.path.splitext(A_table_path)[0]
    matching_npy = [f for f in npy_files if A_base_name in f]
    if matching_npy:
        embedding_input_path = matching_npy[0]
        console.print(f"[yellow]自动选择的嵌入文件: {embedding_input_path}[/yellow]")
        if Prompt.ask("是否使用此嵌入文件？", choices=["y", "n"], default="y") == "n":
            table = Table(title="Available .npy Files")
            table.add_column("Index", style="cyan")
            table.add_column("Filename", style="magenta")
            for i, file in enumerate(npy_files):
                table.add_row(str(i), file)
            console.print(table)
            npy_idx = IntPrompt.ask("请选择 .npy 文件（输入编号）", choices=[str(i) for i in range(len(npy_files))],
                                    default=0)
            embedding_input_path = npy_files[npy_idx]
    else:
        table = Table(title="Available .npy Files")
        table.add_column("Index", style="cyan")
        table.add_column("Filename", style="magenta")
        for i, file in enumerate(npy_files):
            table.add_row(str(i), file)
        console.print(table)
        npy_idx = IntPrompt.ask("请选择 .npy 文件（输入编号）", choices=[str(i) for i in range(len(npy_files))],
                                default=0)
        embedding_input_path = npy_files[npy_idx]

    # 选择 B 表
    table = Table(title="Available .xlsx Files (B Table)")
    table.add_column("Index", style="cyan")
    table.add_column("Filename", style="magenta")
    for i, file in enumerate(xlsx_files):
        table.add_row(str(i), file)
    console.print(table)
    b_file_idx = IntPrompt.ask("请选择 B 表 .xlsx 文件（输入编号）", choices=[str(i) for i in range(len(xlsx_files))],
                               default=0)
    B_table_path = xlsx_files[b_file_idx]

    # 加载数据
    try:
        A_df = pd.read_excel(A_table_path)
        A_embeddings = np.load(embedding_input_path, allow_pickle=True)
        B_df = pd.read_excel(B_table_path)
    except Exception as e:
        console.print(f"[red]加载文件失败: {e}[/red]")
        return

    if len(A_embeddings) != A_df.shape[0]:
        console.print(f"[red]嵌入数量 ({len(A_embeddings)}) 与 A 表行数 ({A_df.shape[0]}) 不匹配！[/red]")
        return

    console.print("A Table Columns:", A_df.columns.tolist())
    console.print("B Table Columns:", B_df.columns.tolist())

    a_col_idx = IntPrompt.ask("请选择 A 表 '物料名称' 所在的列（输入编号）",
                              choices=[str(i) for i in range(len(A_df.columns))], default=0)
    b_col_idx = IntPrompt.ask("请选择 B 表 '物料名称' 所在的列（输入编号）",
                              choices=[str(i) for i in range(len(B_df.columns))], default=0)
    A_col_name = A_df.columns[a_col_idx]
    B_col_name = B_df.columns[b_col_idx]

    # 初始化 B 表输出列
    B_df['MaterialCode'] = pd.Series(dtype='object')
    B_df['UnitCode'] = pd.Series(dtype='object')
    B_df['MatchedMaterialName'] = pd.Series(dtype='object')
    B_df['Similarity'] = pd.Series(dtype='float64')

    # 计算 B 表嵌入
    console.print("计算 B 表嵌入（多线程）...")
    B_texts = [str(text) for text in B_df[B_col_name].fillna("")]
    B_embeddings = compute_embeddings(B_texts, model)
    if B_embeddings is None:
        console.print("[red]B 表嵌入计算失败！[/red]")
        return

    # 多线程匹配
    console.print("匹配 B 表物料（多线程）...")
    with tqdm(total=len(B_embeddings), desc="Matching") as pbar:
        for index, query_embedding in enumerate(B_embeddings):
            similarities = compute_similarities(query_embedding, A_embeddings)
            if similarities is None:
                console.print("[red]相似度计算中断！[/red]")
                return
            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            best_match_row = A_df.iloc[best_match_idx]
            # 检查列名是否存在再赋值
            if '物料编码' in A_df.columns:
                B_df.at[index, 'MaterialCode'] = best_match_row['物料编码']
            if '计量单位编码' in A_df.columns:
                B_df.at[index, 'UnitCode'] = best_match_row['计量单位编码']
            B_df.at[index, 'MatchedMaterialName'] = best_match_row[A_col_name]
            B_df.at[index, 'Similarity'] = best_similarity
            pbar.update(1)

    # 保存结果
    output_path = f"O-{B_table_path}"
    try:
        B_df.to_excel(output_path, index=False)
        console.print(f"[green]处理完成，结果已保存到 {output_path}[/green]")
    except Exception as e:
        console.print(f"[red]保存结果失败: {e}[/red]")


# 主程序
if __name__ == "__main__":
    models = check_ollama_connection()
    if not models:
        console.print("[red]无法继续，请检查 Ollama 服务！[/red]")
        sys.exit(1)

    table = Table(title="Available Models")
    table.add_column("Index", style="cyan")
    table.add_column("ModelName", style="magenta")
    for i, model in enumerate(models):
        table.add_row(str(i), model)
    console.print(Panel.fit(table, title="Model Selection"))

    model_idx = IntPrompt.ask("请选择一个模型（输入编号）", choices=[str(i) for i in range(len(models))], default=0)
    selected_model = models[model_idx]
    console.print(f"[green]已选择模型: {selected_model}[/green]")

    while True:
        console.print("[bold yellow]请选择任务:[/bold yellow]")
        console.print("1. 生成 A 表嵌入 (Process A)")
        console.print("2. 匹配 B 表物料 (Process B)")
        console.print("3. 退出")
        choice = Prompt.ask("输入选项 (1-3)", choices=["1", "2", "3"], default="1")

        if choice == "1":
            process_A(selected_model)
        elif choice == "2":
            process_B(selected_model)
        else:
            console.print("[cyan]感谢使用，再见！[/cyan]")
            sys.exit(0)
