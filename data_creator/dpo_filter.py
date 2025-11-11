#!/usr/bin/env python3
import os
import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util


def load_jsonl(file_path):
    """加载jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(file_path, data):
    """保存jsonl文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def calculate_similarity(chosen, rejected, model):
    """计算chosen和rejected的语义相似度"""
    if not chosen.strip() or not rejected.strip():
        return 0.0
    
    # 编码
    chosen_emb = model.encode(chosen, convert_to_tensor=True, normalize_embeddings=True)
    rejected_emb = model.encode(rejected, convert_to_tensor=True, normalize_embeddings=True)
    
    # 计算余弦相似度
    similarity = util.cos_sim(chosen_emb, rejected_emb).item()
    
    return similarity


def filter_directory(input_dir, output_dir, max_similarity=0.85):
    """
    过滤所有jsonl文件
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        max_similarity: 最大相似度阈值（超过此值会被过滤）
    """
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"最大相似度阈值: {max_similarity}")
    print("=" * 60)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    semantic_model = SentenceTransformer("whaleloops/phrase-bert", device=device)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有jsonl文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"未找到jsonl文件: {input_dir}")
        return
    
    print(f"找到 {len(jsonl_files)} 个文件\n")
    
    total_samples = 0
    total_filtered = 0
    stats = {
        "identical": 0,
        "too_similar": 0,
        "kept": 0
    }
    
    for jsonl_file in jsonl_files:
        print(f"处理文件: {jsonl_file.name}")
        data = load_jsonl(jsonl_file)
        filtered_data = []
        
        file_stats = {"identical": 0, "too_similar": 0, "kept": 0}
        
        for item in tqdm(data, desc=" filtering"):
            total_samples += 1
            
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            
            # 1. 完全相同 -> 过滤
            if chosen.strip() == rejected.strip():
                stats["identical"] += 1
                file_stats["identical"] += 1
                total_filtered += 1
                continue
            
            # 2. 计算语义相似度
            similarity = calculate_similarity(chosen, rejected, semantic_model)
            
            # 3. 相似度过高 -> 过滤
            if similarity > max_similarity:
                stats["too_similar"] += 1
                file_stats["too_similar"] += 1
                total_filtered += 1
                continue
            
            # 4. 保留样本
            filtered_data.append(item)
            stats["kept"] += 1
            file_stats["kept"] += 1
        
        # 保存过滤后的数据
        output_file = output_path / jsonl_file.name
        save_jsonl(output_file, filtered_data)
        
        print(f" 原始: {len(data)} | 保留: {file_stats['kept']} | 过滤: {len(data) - file_stats['kept']}")
        print(f" - 完全相同: {file_stats['identical']}")
        print(f"- 相似度>{max_similarity}: {file_stats['too_similar']}\n")
    
    # 打印总体统计
    print("=" * 60)
    print("过滤统计")
    print("=" * 60)
    print(f"总样本数: {total_samples}")
    print(f"保留样本: {stats['kept']} ({stats['kept']/total_samples*100:.1f}%)")
    print(f"过滤样本: {total_filtered} ({total_filtered/total_samples*100:.1f}%)")
    print("\n过滤分布:")
    print(f"  - 完全相同: {stats['identical']}")
    print(f"  - 相似度>{max_similarity}: {stats['too_similar']}")
    print("=" * 60)
    
    # 保存统计信息
    stats_file = output_path / "filter_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": total_samples,
            "filtered_samples": total_filtered,
            "kept_samples": stats['kept'],
            "filter_rate": total_filtered / total_samples if total_samples > 0 else 0,
            "stats": stats,
            "config": {
                "max_similarity": max_similarity
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n统计信息保存到: {stats_file}")
    print(f"过滤后的数据保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="过滤DPO数据中相似度过高的样本")
    parser.add_argument("--input_dir", type=str, required=True, 
                        help="输入目录（包含DPO jsonl文件）")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="输出目录（保存过滤后的文件）")
    parser.add_argument("--max_similarity", type=float, default=0.85,
                        help="最大相似度阈值，默认0.85（越小过滤越严格）")
    
    args = parser.parse_args()
    
    filter_directory(
        args.input_dir,
        args.output_dir,
        args.max_similarity
    )


if __name__ == "__main__":
    main()