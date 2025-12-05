# 酒店评论情感分析项目

## 项目简介
本项目基于大规模中文预训练模型（Qwen/Qwen2-1.5B-Instruct），针对真实酒店用户评论，开展情感分析与标签提取。项目设计并对比了多种 Prompt 提示策略（零样本、少样本、思维链 CoT），旨在寻找最佳的自动评论理解与分析方案。模型会自动读取上千条真实评论，输出每条评论的情感（正面/负面）和核心关键词标签，并将最终分析结果保存为 CSV 文件，方便后续数据分析与可视化。

## 项目特色
- **多种 Prompt 策略对比**：支持零样本、少样本、思维链（Chain-of-Thought）等模板，便于实验不同提示方法对结果的影响。
- **适配国产开源大模型**：无须联网，即可在本地完成酒店评论批量理解。
- **批量处理与结果可视化**：支持一键批量处理评论，并用 CSV 文件清晰存储结果，便于后续进一步分析。

## 依赖环境
- Python 3.8 及以上
- 详见 `requirements.txt`，核心依赖包括：
  - transformers
  - torch
  - pandas
  - tqdm

## 快速开始

1. **克隆本项目**
   ```
   git clone https://github.com/Douyh123/hotel-rag.git
   cd whye
   ```

2. **安装依赖**
   ```
   pip install -r requirements.txt
   ```

3. **准备数据集**
   - 将数据集（如 ChnSentiCorp_htl_all.csv）放入 `./data/` 目录下。

4. **运行实验脚本**
   ```
   python src/prompt_experiment.py
   ```

5. **查看结果**
   - 结果将保存在 `./output/prompt_experiment_results.csv`，包含每条评论的情感与标签，并附有不同提示策略下的输出。

## 目录结构

```
.
├── data/                                  # 数据集存放文件夹
├── output/                                # 结果输出文件夹
├── src/
│   ├── prompt_experiment.py               # 主实验脚本，推荐运行
│   └── prompt_demo.py                     # 单条 demo 预测脚本
├── requirements.txt                       # 依赖包列表
├── README.md
```

## 常见问题
- **模型加载失败/缓慢？** 请确保已正确安装 `transformers`，并本地网络通畅或预先下载模型。
- **显存不足？** 支持自动释放 CUDA 显存。处理大数据可适当减少批处理数。
- **输出乱码？** 结果文件已使用 `utf-8-sig` 编码，确保用合适工具（如 Excel）打开。

## 参考
- [Qwen 大模型 (阿里)](https://huggingface.co/Qwen)
- [ChnSentiCorp 数据集](https://github.com/SophonPlus/ChineseNlpCorpus)
- transformers / pytorch / Huggingface 社区

欢迎交流、star 和提出改进建议！