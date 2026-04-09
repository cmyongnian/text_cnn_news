# text_cnn_news
数据挖掘A实验三
# TextCNN News Classification

基于 `PyTorch` 的中文新闻文本分类项目，完成“文本数据挖掘”实验要求。

本项目包含两个实验：

- **实验一（exp1）**：仅使用 `sentence` 进行文本分类
- **实验二（exp2）**：同时使用 `sentence + keywords` 进行文本分类

模型均为 **CNN 变种**：
- `TextCNN`：实验一基线模型
- `KeywordFusionTextCNN`：实验二关键词融合模型

---

## 1. 项目目标

根据给定新闻分类数据集，完成多分类任务，并比较以下两种方案：

### 实验一：只使用文本
- 输入：`sentence`
- 模型：`TextCNN`
- 目的：建立文本分类基线

### 实验二：文本 + 关键词
- 输入：`sentence + keywords`
- 模型：`KeywordFusionTextCNN`
- 目的：验证关键词信息是否能提升分类效果

---

## 2. 项目结构

```text
text_cnn_news/
├─ README.md
└─ textcnn_project/
   ├─ config.yaml
   ├─ preprocess.py
   ├─ train.py
   ├─ utils.py
   ├─ model_textcnn.py
   ├─ model_textcnn_fusion.py
   ├─ data/
   │  ├─ labels.json
   │  ├─ train.json
   │  ├─ test.json
   │  └─ processed/
   │     ├─ vocab.json
   │     ├─ train_tokens.json
   │     ├─ test_tokens.json
   │     ├─ train_seq.pt
   │     ├─ test_seq.pt
   │     ├─ train_key_mask.pt
   │     ├─ test_key_mask.pt
   │     └─ data_statistics.txt
   └─ outputs/
      ├─ models/
      │  ├─ best_model_exp1.pth
      │  └─ best_model_exp2.pth
      ├─ logs/
      │  ├─ exp1_log.txt
      │  └─ exp2_log.txt
      ├─ figs/
      │  ├─ exp1_loss_curve.png
      │  └─ exp2_loss_curve.png
      └─ reports/
         ├─ classification_report_exp1.txt
         ├─ classification_report_exp2.txt
         ├─ model_summary.txt
         └─ results_comparison.csv