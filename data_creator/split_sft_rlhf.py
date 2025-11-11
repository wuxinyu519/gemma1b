import json
import random
from pathlib import Path

random.seed(42)

# 定义目录
input_dir = Path("augmented_outputs")
sft_dir = Path("sft_data")
rlhf_dir = Path("rlhf_data")
test_dir = Path("test_data")

# 创建输出目录
sft_dir.mkdir(exist_ok=True)
rlhf_dir.mkdir(exist_ok=True)
test_dir.mkdir(exist_ok=True)

# 统计信息
total_sft = 0
total_rlhf = 0
total_test = 0

# 遍历所有jsonl文件
for jsonl_file in input_dir.glob("*.jsonl"):
    print(f"处理文件: {jsonl_file.name}")
    
    # 读取所有数据
    data = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    print(f"  总数据量: {len(data)}")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 计算分割点 (60% SFT, 38% RLHF, 2% Test)
    sft_end = int(len(data) * 0.6)
    rlhf_end = int(len(data) * 0.98) 
    
    sft_data = data[:sft_end]
    rlhf_data = data[sft_end:rlhf_end]
    test_data = data[rlhf_end:]
    
    # 保存SFT数据
    sft_filename = sft_dir / f"sft_{jsonl_file.name}"
    with open(sft_filename, 'w', encoding='utf-8') as f:
        for item in sft_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存RLHF数据
    rlhf_filename = rlhf_dir / f"rlhf_{jsonl_file.name}"
    with open(rlhf_filename, 'w', encoding='utf-8') as f:
        for item in rlhf_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存Test数据
    test_filename = test_dir / f"test_{jsonl_file.name}"
    with open(test_filename, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    total_sft += len(sft_data)
    total_rlhf += len(rlhf_data)
    total_test += len(test_data)
    
    print(f"  SFT:  {len(sft_data)} 条 ({len(sft_data)/len(data)*100:.1f}%) → {sft_filename.name}")
    print(f"  RLHF: {len(rlhf_data)} 条 ({len(rlhf_data)/len(data)*100:.1f}%) → {rlhf_filename.name}")
    print(f"  Test: {len(test_data)} 条 ({len(test_data)/len(data)*100:.1f}%) → {test_filename.name}")
    print()

print("=" * 60)
print(f"总计 SFT:  {total_sft} 条")
print(f"总计 RLHF: {total_rlhf} 条")
print(f"总计 Test: {total_test} 条")
print(f"比例: SFT {total_sft/(total_sft+total_rlhf+total_test)*100:.1f}% | RLHF {total_rlhf/(total_sft+total_rlhf+total_test)*100:.1f}% | Test {total_test/(total_sft+total_rlhf+total_test)*100:.1f}%")