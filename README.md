# Gemma1B RLHF/SFT Pipeline

## DIR
```
gemma1b/
│
├── data_creator/                 # 数据生成与预处理脚本
│
├── eval/                         # 测试与评估模块
│   ├── infinite_bench/           # InfiniteBench 评测
│   ├── final_infer.py            # 推理主脚本（需根据 template 调整路径）
│   └── infer.sh                  # 执行推理脚本
│
├── sft/                          # SFT (Supervised Fine-tuning) 模块
│   ├── data_loader.py
│   ├── sft_trainer.sh            # 训练启动脚本
│   ├── train.py / test.py        # 主训练与测试脚本
│   └── <checkpoint>/               # 保存模型与评估结果
│
├── rlhf/                         # RLHF / DPO 模块
    ├── dpo_trainer.sh            # DPO 训练脚本
    ├── tune_w_rlhf.py            # RLHF 调优主脚本
    └── <checkpoint>/               # 保存模型与评估结果

```

---

## 数据准备

### 1️下载数据集  
将infinitebench数据下载到 `eval/` 目录下，结构如下：

```
eval/
├── infinite_bench/
│   ├── infinitebench_gpt_gt/
│   │   ├── cleaned_tags_jsonl/
│   │   │   └── individual_files/
│   │   ├── gpt_groundtruth_gpt-4o/
│   │   └── gpt_groundtruth_gpt-4o_change/
```

### 2放置模型 checkpoint  
- SFT 模型的 checkpoint 放在：
  ```
  sft/
  ```
- RLHF 模型的 checkpoint 放在：
  ```
  rlhf/
  ```

---

## 测试（Evaluation）

若要在 **InfiniteBench** 上测试模型性能：

```bash
cd eval
qsub infer.sh
```

共有三种模板 (Template) 可用于测试，每次需在 `final_infer.py` 中手动切换对应路径：

| 模板编号 | Ground Truth 路径 | 说明 |
|-----------|--------------------|------|
| Template 1 | `./infinite_bench/infinitebench_gpt_gt/cleaned_tags_jsonl/individual_files` | 默认 GT 结构 |
| Template 2 | `./infinite_bench/infinitebench_gpt_gt/gpt_groundtruth_gpt-4o` | GPT-4o 基准标签 |
| Template 3 | `./infinite_bench/infinitebench_gpt_gt/gpt_groundtruth_gpt-4o_change` | GPT-4o 修改版本 |

---

## 训练阶段

### SFT 训练

```bash
cd sft
qsub sft_trainer.sh
```

训练完成后，结果将自动保存在：

```
sft/<checkpoint>/evaluation result/
```

---

### RLHF（DPO）训练

```bash
cd rlhf
qsub dpo_trainer.sh
```

训练完成后，结果保存在：

```
rlhf/<checkpoint>/evaluationresult/
```

---

## 备注

- 所有脚本默认基于 **Gemma1B** 模型，可在各 `.sh` 文件中修改模型路径或训练参数。
- 训练与推理日志默认输出到各目录下的 `.out` 文件。
