#!/usr/bin/env python3
import os
import json
import re
from tqdm import tqdm
import argparse
import time
import glob
from openai import OpenAI

# ============================================================================
# GPT API 调用部分
# ============================================================================

def init_openai_client(api_key=None, base_url=None):
    """初始化 OpenAI 客户端"""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if api_key is None:
        raise ValueError("请设置 OPENAI_API_KEY 环境变量或通过参数传入")
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url  # 如果使用代理或其他 API endpoint
    )
    return client

def truncate_context(context: str, max_chars: int = 1500) -> str:
    """保留前后部分内容"""
    if len(context) <= max_chars:
        return context
    
    half = max_chars // 2
    return context[:half] + "\n\n[Content truncated]\n\n" + context[-half:]

def extract_tags_with_explanations(tags_text):
    """提取标签和解释"""
    try:
        # 尝试解析 JSON 数组
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, tags_text, re.DOTALL)
        
        if json_matches:
            json_str = json_matches[-1]
            try:
                parsed_json = json.loads(json_str)
                if isinstance(parsed_json, list):
                    valid_tags = []
                    for item in parsed_json:
                        if isinstance(item, dict) and "tag" in item and "explanation" in item:
                            valid_tags.append({
                                "tag": str(item["tag"]).strip(),
                                "explanation": str(item["explanation"]).strip()
                            })
                    if valid_tags:
                        return valid_tags
            except json.JSONDecodeError:
                pass
        
        # 尝试解析单个 JSON 对象
        single_json_pattern = r'\{[^{}]*"tag"[^{}]*"explanation"[^{}]*\}'
        single_matches = re.findall(single_json_pattern, tags_text)
        
        if single_matches:
            valid_tags = []
            for match in single_matches:
                try:
                    item = json.loads(match)
                    if "tag" in item and "explanation" in item:
                        valid_tags.append({
                            "tag": str(item["tag"]).strip(),
                            "explanation": str(item["explanation"]).strip()
                        })
                except:
                    continue
            if valid_tags:
                return valid_tags
        
        return [{"tag": "Parse_Error", "explanation": "Unable to parse response"}]
        
    except Exception as e:
        print(f"Error extracting tags: {e}")
        return []

def call_gpt_api(client, prompt, model="gpt-4o-mini", max_tokens=512, temperature=0.7):
    """调用 GPT API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates tags for user queries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"API call error: {e}")
        return None

def run_inference(client, data, output_file, model="gpt-4o-mini", save_interval=50, sleep_time=0.5):
    """使用 GPT API 生成 ground truth"""
    print(f"Running GPT inference on {len(data)} samples using model: {model}...")
    
    start_time = time.time()
    all_results = []
    
    # 定义 prompt 模板
    prompt_template = """You are a helpful assistant. Please identify tags of user intentions in the following user query and provide an explanation for each tag. Only select tags from the following tag list. Please respond in the JSON format {{"tag": str, "explanation": str}}.

## Tag list
- Tag name: Code.Debug, Description: Finding which function in a code repo contains a crashing error (in multiple choice form).
- Tag name: Code.Run, Description: Simulating execution of multiple simple, synthetic functions.
- Tag name: Retrieve.KV, Description: Finding the corresponding value and a value from a JSON dictionary.
- Tag name: En.MutipleChoice, Description: Multiple choice questions regarding a given book.
- Tag name: Zh.QA, Description: Question answering on a set of newly collected books in Chinese.
- Tag name: En.QA, Description: Free-form question answering based on the fake book in English.
- Tag name: En.Sum, Description: Summarization of a English book.
- Tag name: En.Dia, Description: Identification of talkers in partially anonymized dialogue scripts in English.
- Tag name: Math.Calc, Description: Calculations involving super-long arithmetic equations and output intermediate results of the given numerical expression including + and - arithmetic.
- Tag name: Math.Find, Description: Finding largest integers in a given numerical list.
- Tag name: Retrieve.Number, Description: Locating repeated hidden numbers in a noisy long context.
- Tag name: Retrieve.PassKey, Description: Retrieving hidden keys in a noisy long context.

Query: {query}"""
    
    for i, item in enumerate(tqdm(data, desc="Generating ground truth")):
        # 准备输入 - 使用 input/prompt + context 的逻辑
        if 'input' in item:
            input_content = item['input']
        elif 'prompt' in item:
            input_content = item['prompt']
        else:
            input_content = ''
            print(f"Warning: Sample {i} missing 'input' or 'prompt' field")
        
        context_content = item.get('context', '')
        
        if context_content:
            inference_text = f"{input_content}\n\n{context_content}"
        else:
            inference_text = input_content
        
        truncated_context = truncate_context(inference_text, max_chars=1500)
        
        prompt = prompt_template.format(query=truncated_context)
        
        # 调用 API
        response = call_gpt_api(client, prompt, model=model)
        
        if response:
            # 解析响应
            tags = extract_tags_with_explanations(response)
            
            result = {
                'id': item.get('id', f'sample_{i}'),
                'input': item.get('input', ''),
                'prompt': item.get('prompt', ''),
                'context': item.get('context', ''),
                'inference_context': inference_text,
                'raw_response': response,
                'ground_truth': tags  # 生成的标签作为 ground truth
            }
        else:
            result = {
                'id': item.get('id', f'sample_{i}'),
                'input': item.get('input', ''),
                'prompt': item.get('prompt', ''),
                'context': item.get('context', ''),
                'inference_context': inference_text,
                'raw_response': None,
                'ground_truth': []
            }
        
        all_results.append(result)
        
        # 定期保存
        if (i + 1) % save_interval == 0:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)
                print(f"Checkpoint saved: {i+1}/{len(data)} samples")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")
        
        # API 限流
        time.sleep(sleep_time)
    
    # 最终保存
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving final results: {e}")
    
    elapsed = time.time() - start_time
    print(f"Generation completed in {elapsed:.2f} seconds")
    print(f"Average time per sample: {elapsed/len(data):.2f} seconds")
    
    return all_results

# ============================================================================
# 数据加载和处理
# ============================================================================

def find_json_files(directory):
    """查找所有 JSON 文件"""
    patterns = ['*.json', '*.jsonl']
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(directory, pattern)))
    return sorted(files)

def load_data(file_path, num_samples=None):
    """加载数据"""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.jsonl'):
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            else:
                content = json.load(f)
                if isinstance(content, list):
                    data = content
                elif isinstance(content, dict):
                    data = [content]
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []
    
    if num_samples and num_samples < len(data):
        data = data[:num_samples]
    
    return data

def process_single_file(client, json_file, output_dir, args):
    """处理单个文件"""
    file_basename = os.path.basename(json_file)
    print(f"Processing: {file_basename}")
    
    # 加载数据
    data = load_data(json_file, args.num_samples)
    if not data:
        print(f"No data loaded from {file_basename}")
        return None
    
    print(f"Loaded {len(data)} samples")
    
    # 设置输出文件
    base_name = os.path.splitext(file_basename)[0]
    output_file = os.path.join(output_dir, f"{base_name}_groundtruth.json")
    
    # 运行推理生成 ground truth
    results = run_inference(
        client, data, output_file,
        model=args.model,
        save_interval=args.save_interval,
        sleep_time=args.sleep_time
    )
    
    if not results:
        print("Generation produced no results")
        return None
    
    print(f"✓ Generated ground truth for {len(results)} samples")
    
    return {
        'file_name': file_basename,
        'num_samples': len(results),
        'output_file': output_file
    }

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GPT-based ground truth generation")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .json/.jsonl files")
    parser.add_argument("--output_prefix", type=str, default="gpt_groundtruth",
                        help="Prefix for output directory")
    parser.add_argument("--model", type=str, default="gpt-4o-mini",
                        choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                        help="GPT model to use")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--base_url", type=str, default=None,
                        help="Custom API base URL (optional)")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit samples per file (default: process all)")
    parser.add_argument("--save_interval", type=int, default=50,
                        help="Save interval for incremental saves")
    parser.add_argument("--sleep_time", type=float, default=0.5,
                        help="Sleep time between API calls (seconds)")

    args = parser.parse_args()
    
    # 验证输入
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return
    
    # 初始化 OpenAI 客户端
    try:
        client = init_openai_client(args.api_key, args.base_url)
        print("✓ OpenAI client initialized successfully")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return
    
    # 设置输出目录
    output_dir = f"{args.output_prefix}_{args.model.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Sleep time: {args.sleep_time}s")
    
    # 查找文件
    json_files = find_json_files(args.data_dir)
    if not json_files:
        print("No JSON files found!")
        return
    
    print(f"\nFound {len(json_files)} files to process")
    
    # 处理所有文件
    all_file_results = []
    
    for i, json_file in enumerate(json_files, 1):
        print(f"\n{'='*80}")
        print(f"FILE {i}/{len(json_files)}")
        print(f"{'='*80}")
        
        result = process_single_file(client, json_file, output_dir, args)
        
        if result:
            all_file_results.append(result)
    
    # 汇总结果
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {len(all_file_results)}/{len(json_files)}")
    
    if all_file_results:
        print(f"\nPROCESSED FILES:")
        print(f"{'-'*80}")
        for result in all_file_results:
            print(f"{result['file_name']:50s} | Samples: {result['num_samples']:4d}")
        
        total_samples = sum([r['num_samples'] for r in all_file_results])
        print(f"\nTotal ground truth samples generated: {total_samples}")
        
        # 保存汇总信息
        summary_file = os.path.join(output_dir, "generation_summary.json")
        summary_data = {
            'model': args.model,
            'total_files': len(all_file_results),
            'total_samples': total_samples,
            'files': [
                {
                    'file_name': r['file_name'],
                    'num_samples': r['num_samples'],
                    'output_file': r['output_file']
                }
                for r in all_file_results
            ]
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            print(f"\nSummary saved to: {summary_file}")
        except Exception as e:
            print(f"Error saving summary: {e}")
    
    print(f"\n{'='*80}")
    print("GENERATION COMPLETED!")
    print(f"{'='*80}")
    print(f"All results saved in: {output_dir}")

if __name__ == "__main__":
    main()