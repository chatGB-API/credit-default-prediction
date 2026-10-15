\# 信用违约预测（Credit Default Prediction）



\## 1. 问题定义

预测客户未来两年内是否会发生信用违约（二分类）。

\- 输入：客户特征（年龄、月收入、信用卡利用率、逾期次数等）

\- 输出：违约概率 + 风险等级（高/中/低）

\- 目标：在控制坏账率的前提下，尽量识别高风险客户。



\## 2. 数据来源

\- 数据集：Kaggle Give Me Some Credit

\- 文件：data/raw/cs-training.csv

\- 样本量：约 150,000 行

\- 目标列：SeriousDlqin2yrs（1=违约，0=未违约）

\- 类别比例：违约约 6.7%，属于类别不平衡。



\## 3. 方法

1\. 数据清洗：缺失值填充、异常值截断

2\. 特征工程：TotalPastDue、IncomePerPerson、HasRealEstate、LatePaymentRatio 等

3\. 基线模型：逻辑回归

4\. 主模型：LightGBM

5\. 模型解释：SHAP 全局 + 局部解释

6\. 部署：FastAPI 接口 + Streamlit 网页



\## 4. 结果

| 模型 | AUC | PR-AUC | F1 |

|---|---|---|---|

| 逻辑回归（基线） | 0.85431 | 0.37863 | 0.25197 |

| LightGBM（原始特征） |  0.86724 |  0.39510 | 0.29172 |

| LightGBM + 特征工程 | 0.86717 |0.39260| 0.28719 |



> 注：以上为示例，请替换成你实际跑出来的结果。



\## 5. 怎么运行

\### 安装依赖

```bash

pip install -r requirements.txt

