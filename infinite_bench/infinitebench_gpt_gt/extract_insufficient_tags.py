#!/usr/bin/env python3
"""
提取JSONL文件中tag数量小于5的数据的truncated_input字段
保存结果到CSV文件，包含来源文件信息
"""
import json
import os
import glob
import csv
from typing import List, Dict, Any
from pathlib import Path

class TagInsufficientExtractor:
    def __init__(self, directory_path: str):
        """
        初始化提取器
        Args:
            directory_path: 包含JSONL文件的目录路径
        """
        self.directory_path = directory_path
        self.jsonl_files = self.find_jsonl_files()
        self.results = []
        
    def find_jsonl_files(self) -> List[str]:
        """查找目录中的所有JSONL文件"""
        jsonl_files = []
        
        if os.path.isdir(self.directory_path):
            # 查找当前目录的JSONL文件
            pattern = os.path.join(self.directory_path, "*.jsonl")
            jsonl_files.extend(glob.glob(pattern))
            
            # 递归查找子目录的JSONL文件
            for root, dirs, files in os.walk(self.directory_path):
                for file in files:
                    if file.endswith('.jsonl'):
                        file_path = os.path.join(root, file)
                        if file_path not in jsonl_files:
                            jsonl_files.append(file_path)
        
        return sorted(jsonl_files)
    
    def parse_tags(self, parsed_tags) -> List[Dict]:
        """解析parsed_tags字段"""
        try:
            if isinstance(parsed_tags, list):
                return parsed_tags
            elif isinstance(parsed_tags, str):
                # 尝试解析JSON字符串
                if parsed_tags.strip().startswith('['):
                    return json.loads(parsed_tags)
                else:
                    # 尝试eval解析
                    import ast
                    return ast.literal_eval(parsed_tags)
            else:
                return []
        except Exception as e:
            print(f"解析标签失败: {e}")
            return []
    
    def count_tags(self, parsed_tags) -> int:
        """计算标签数量"""
        tags = self.parse_tags(parsed_tags)
        if not isinstance(tags, list):
            return 0
        
        # 计算有效的tag数量
        valid_count = 0
        for item in tags:
            if isinstance(item, dict) and 'tag' in item and item['tag'].strip():
                valid_count += 1
        
        return valid_count
    
    def extract_insufficient_data(self):
        """提取tag个数小于5的数据的truncated_input"""
        print(f"🔍 开始检查 {len(self.jsonl_files)} 个JSONL文件...")
        
        total_found = 0
        
        for file_path in self.jsonl_files:
            filename = os.path.basename(file_path)
            print(f"\n📁 检查文件: {filename}")
            
            try:
                file_count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        
                        try:
                            data = json.loads(line)
                            if 'parsed_tags' in data:
                                tag_count = self.count_tags(data['parsed_tags'])
                                if tag_count < 5:
                                    # 提取truncated_input字段
                                    truncated_input = data.get('truncated_input', '')
                                    
                                    # 提取parsed_tags内容
                                    parsed_tags_content = data.get('parsed_tags', [])
                                    
                                    # 保存结果
                                    self.results.append({
                                        'source_file': filename,
                                        'line_number': line_num,
                                        'tag_count': tag_count,
                                        'truncated_input': truncated_input,
                                        'parsed_tags': parsed_tags_content
                                    })
                                    
                                    file_count += 1
                                    total_found += 1
                        
                        except json.JSONDecodeError as e:
                            print(f"   ⚠️  第{line_num}行JSON解析错误: {e}")
                            continue
                
                if file_count > 0:
                    print(f"   ❌ 发现 {file_count} 条tag不足的数据")
                else:
                    print(f"   ✅ 所有数据tag数量都>=5")
                    
            except Exception as e:
                print(f"   ❌ 文件读取失败: {e}")
        
        print(f"\n📊 总计发现 {total_found} 条tag不足的数据")
        return total_found
    
    def save_to_csv(self, output_file: str = "tag_insufficient_data.csv"):
        """保存结果到CSV文件"""
        if not self.results:
            print("❌ 没有数据需要保存")
            return
        
        print(f"\n💾 保存结果到: {output_file}")
        
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['source_file', 'line_number', 'tag_count', 'truncated_input', 'parsed_tags']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # 写入表头
                writer.writeheader()
                
                # 写入数据
                for result in self.results:
                    # 为CSV格式转换parsed_tags为字符串
                    csv_result = result.copy()
                    csv_result['parsed_tags'] = json.dumps(result['parsed_tags'], ensure_ascii=False)
                    writer.writerow(csv_result)
            
            print(f"✅ 成功保存 {len(self.results)} 条记录")
            print(f"📁 文件路径: {os.path.abspath(output_file)}")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def save_to_json(self, output_file: str = "tag_insufficient_data.json"):
        """保存结果到JSON文件"""
        if not self.results:
            print("❌ 没有数据需要保存")
            return
        
        print(f"\n💾 保存结果到: {output_file}")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 成功保存 {len(self.results)} 条记录")
            print(f"📁 文件路径: {os.path.abspath(output_file)}")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")
    
    def display_summary(self):
        """显示统计摘要"""
        if not self.results:
            return
        
        print(f"\n📊 统计摘要:")
        
        # 按文件统计
        file_stats = {}
        for result in self.results:
            filename = result['source_file']
            if filename not in file_stats:
                file_stats[filename] = 0
            file_stats[filename] += 1
        
        print(f"   各文件统计:")
        for filename, count in sorted(file_stats.items()):
            print(f"     {filename}: {count} 条")
        
        # 按tag数量统计
        tag_count_stats = {}
        for result in self.results:
            tag_count = result['tag_count']
            if tag_count not in tag_count_stats:
                tag_count_stats[tag_count] = 0
            tag_count_stats[tag_count] += 1
        
        print(f"   按tag数量统计:")
        for tag_count, count in sorted(tag_count_stats.items()):
            print(f"     {tag_count}个tag: {count} 条")
        
        # 显示几个示例
        print(f"\n📝 数据示例:")
        for i, result in enumerate(self.results[:3], 1):
            truncated_preview = result['truncated_input'][:100] + "..." if len(result['truncated_input']) > 100 else result['truncated_input']
            print(f"   {i}. 文件: {result['source_file']}, 行号: {result['line_number']}, tag数: {result['tag_count']}")
            print(f"      truncated_input: {truncated_preview}")
            
            # 显示parsed_tags内容
            tags = self.parse_tags(result['parsed_tags'])
            print(f"      parsed_tags:")
            for j, tag in enumerate(tags, 1):
                if isinstance(tag, dict) and 'tag' in tag:
                    explanation = tag.get('explanation', '无解释')
                    tag_preview = explanation[:50] + "..." if len(explanation) > 50 else explanation
                    print(f"        {j}. {tag['tag']} - {tag_preview}")
                else:
                    print(f"        {j}. {tag}")
            print()
    
    def run(self):
        """运行提取器"""
        print("=" * 60)
        print("🏷️  JSONL文件tag不足数据提取器")
        print("=" * 60)
        print(f"📁 目标目录: {self.directory_path}")
        print(f"📄 找到文件: {len(self.jsonl_files)} 个")
        
        if not self.jsonl_files:
            print("❌ 没有找到JSONL文件")
            return
        
        # 提取数据
        total_found = self.extract_insufficient_data()
        
        if total_found > 0:
            # 显示摘要
            self.display_summary()
            
            # 保存结果
            self.save_to_csv()
            self.save_to_json()
            
            print(f"\n🎉 提取完成！")
            print(f"   - CSV格式: tag_insufficient_data.csv (包含truncated_input和parsed_tags)")
            print(f"   - JSON格式: tag_insufficient_data.json (包含完整结构化数据)")
            print(f"\n💡 说明:")
            print(f"   - CSV中的parsed_tags字段为JSON字符串格式")
            print(f"   - JSON文件保持原始的列表/字典结构")
        else:
            print(f"\n🎉 所有文件的数据tag数量都>=5，无需提取！")

def main():
    """主函数"""
    # 设置目录路径
    directory_path = "./"
    
    if not directory_path:
        directory_path = "./"  # 默认目录
        print(f"使用默认目录: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"❌ 目录不存在: {directory_path}")
        return
    
    # 创建提取器并运行
    extractor = TagInsufficientExtractor(directory_path)
    extractor.run()

if __name__ == "__main__":
    main()