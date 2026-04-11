# TextCNN News Classification

基于 PyTorch 的中文新闻文本分类课程实验项目。

本项目完成“文本数据挖掘”实验要求，包含两个实验：

- **实验一（exp1）**：仅使用 `sentence` 进行文本分类
- **实验二（exp2）**：同时使用 `sentence + keywords` 进行文本分类

两个实验都只使用 **CNN 变种模型**，这里采用：

- `TextCNN`：实验一基线模型
- `KeywordFusionTextCNN`：实验二关键词融合模型

根据实验要求：
- 实验一只能使用文本数据中的 `sentence`
- 实验二需要同时使用 `sentence` 和 `keywords`
- 模型只能使用 CNN 变种，例如 TextCNN
- 可以使用 `jieba` 等工具进行预处理，但不能引入新数据或其他数据集。:contentReference[oaicite:2]{index=2}

---
# 2. 环境配置

conda create -n textcnn python=3.10 -y
conda activate textcnn
pip install torch jieba pyyaml matplotlib scikit-learn numpy

将原始数据放到：

textcnn_project/data/raw/
├─ labels.json
├─ train.json
└─ test.json

如果你把数据放在别的位置，请同步修改 config.yaml 中的：
paths:
  raw_dir: data/raw

# 复现步骤

cd textcnn_project
python preprocess.py

第二步：运行实验一
python train.py --exp exp1
第三步：运行实验二
python train.py --exp exp2

第四步：运行数据分析
python analyze_results.py --exp exp1
python analyze_results.py --exp exp2


## 1. 项目目标

给定新闻文本分类数据集，完成多分类任务，并比较以下两种方案：

### 实验一：只使用文本
- 输入：`sentence`
- 模型：`TextCNN`
- 作用：建立基线分类效果

### 实验二：文本 + 关键词
- 输入：`sentence + keywords`
- 模型：`KeywordFusionTextCNN`
- 作用：验证关键词信息是否有助于提升分类效果

---

## 2. 数据说明

数据集包含三个文件：

- `labels.json`：类别编号与类别描述
- `train.json`：训练集
- `test.json`：测试集

其中 `train.json` 和 `test.json` 每条样本包含：
- `label`：类别编号
- `label_desc`：类别描述
- `sentence`：新闻文本
- `keywords`：关键词。:contentReference[oaicite:3]{index=3}

---

## 3. 项目结构

推荐使用以下目录结构：

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
   │  ├─ raw/
   │  │  ├─ labels.json
   │  │  ├─ train.json
   │  │  └─ test.json
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