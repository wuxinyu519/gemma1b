#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from vllm import LLM, SamplingParams

def load_jsonl(file_path):
    """加载单个jsonl文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sample = json.loads(line)
                if "input" in sample and "output" in sample:
                    data.append(sample)
            except json.JSONDecodeError:
                continue
    return data


def save_jsonl(file_path, data):
    """保存jsonl文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def generate_rejected_batch(llm, prompts, template, sampling_params):
    """批量生成rejected回答"""
    # 构造完整prompts
    full_prompts = [template.format(query=p) for p in prompts]
    
    # vLLM批量生成
    outputs = llm.generate(full_prompts, sampling_params)
    
    # 提取回答
    responses = []
    for output in outputs:
        generated_text = output.outputs[0].text.strip()
        
        # 提取Assistant回答部分
        if "Assistant:" in generated_text:
            response = generated_text.split("Assistant:", 1)[-1].strip()
        else:
            response = generated_text
        
        responses.append(response)
    
    return responses


def main():
    parser = argparse.ArgumentParser(description="Generate DPO dataset using vLLM (batch inference)")
    parser.add_argument("--rlhf_data_dir", type=str, default="rlhf_data")
    parser.add_argument("--base_model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="./dpo_data")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--limit_data", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===============================
    # 加载vLLM模型
    # ===============================
    print("=" * 60)
    print(f"Step 1: 加载vLLM模型: {args.base_model_name}")
    print("=" * 60)
    
    llm = LLM(
        model=args.base_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
    )
    
    # 采样参数
    sampling_params = SamplingParams(
        temperature=1.2,
        top_p=0.9,
        max_tokens=256,
        repetition_penalty=1.2,
    )
    
    template = (
    "<start_of_turn>user\n"
    "{query}\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)
    
    # ===============================
    # 遍历所有RLHF文件
    # ===============================
    print("\n" + "=" * 60)
    print("Step 2: 处理RLHF数据文件")
    print("=" * 60)
    
    rlhf_dir = Path(args.rlhf_data_dir)
    rlhf_files = list(rlhf_dir.glob("*.jsonl"))
    
    if not rlhf_files:
        print(f"没有找到jsonl文件: {args.rlhf_data_dir}")
        return
    
    print(f"找到 {len(rlhf_files)} 个文件")
    
    total_processed = 0
    
    for rlhf_file in rlhf_files:
        print(f"\n处理文件: {rlhf_file.name}")
        
        # 加载数据
        data = load_jsonl(rlhf_file)
        
        # 限制数据量
        if args.limit_data and args.limit_data < len(data):
            import random
            random.seed(42)
            data = random.sample(data, args.limit_data)
        
        print(f"   数据量: {len(data)}")
        
        # 批量生成DPO数据
        dpo_data = []
        
        for i in tqdm(range(0, len(data), args.batch_size), desc=f"  Generating"):
            batch = data[i:i + args.batch_size]
            
            # 提取prompts和chosen
            prompts = [item["input"] for item in batch]
            chosens = [item["output"] for item in batch]
            
            # 批量生成rejected
            rejecteds = generate_rejected_batch(llm, prompts, template, sampling_params)
            
            # 构造DPO数据
            for prompt, chosen, rejected in zip(prompts, chosens, rejecteds):
                dpo_data.append({
                    "input": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
        
        # 保存到输出目录，保持原文件名
        output_file = Path(args.output_dir) / rlhf_file.name
        save_jsonl(output_file, dpo_data)
        
        print(f"  保存: {len(dpo_data)} 条 → {output_file}")
        total_processed += len(dpo_data)
    
    # ===============================
    # 保存meta信息
    # ===============================
    meta = {
        "base_model": args.base_model_name,
        "total_files": len(rlhf_files),
        "total_samples": total_processed,
        "batch_size": args.batch_size,
        "source_dir": args.rlhf_data_dir,
    }
    meta_file = os.path.join(args.output_dir, "meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("DPO数据生成")
    print("=" * 60)
    print(f"输出目录: {args.output_dir}")
    print(f"总计: {total_processed} 条数据，{len(rlhf_files)} 个文件")
   

if __name__ == "__main__":
    main()


