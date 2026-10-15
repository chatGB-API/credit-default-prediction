import pandas as pd
import numpy as np
from src.preprocess import load_and_clean, engineer_features, clean_column_names

df = load_and_clean('data/raw/cs-training.csv')
df = engineer_features(df)
df = clean_column_names(df)

from sklearn.model_selection import train_test_split, StratifiedKFold

X = df.drop('SeriousDlqin2yrs', axis=1)
Y = df['SeriousDlqin2yrs']
#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


# print(X_train.columns.tolist())

#训练基线模型
from sklearn.linear_model import LogisticRegression      # 逻辑回归模型
from sklearn.preprocessing import StandardScaler          # 标准化
from sklearn.pipeline import make_pipeline                # 把多个步骤串成流水线
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score  # 三个评估指标

baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
baseline.fit(X_train, y_train)
pred_proba = baseline.predict_proba(X_test)[:,1]

# print('base_line_AUC:',roc_auc_score(y_test,pred_proba))
# print('base_line_PR-AUC:',average_precision_score(y_test,pred_proba))
# print('base_line_F1:',f1_score(y_test,(pred_proba>0.5).astype(int)))


#训练 LightGBM

import lightgbm as lgb

#k折交叉验证
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X_train))
models = []


for fold, (train_index, test_index) in enumerate(skf.split(X_train, y_train)):
    X_tr,y_tr = X_train.iloc[train_index], y_train.iloc[train_index]
    X_va,y_va = X_train.iloc[test_index], y_train.iloc[test_index]

    model_lightgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        bagging_seed=42,
        feature_fraction_seed=42,
        verbose=-1
    )

    model_lightgbm.fit(X_tr, y_tr,
                       eval_set=[(X_va, y_va)],
                       eval_metric='auc',
                       callbacks=[lgb.early_stopping(50),lgb.log_evaluation(100)]
                       )

    oof_preds[test_index] = model_lightgbm.predict_proba(X_va)[:,1]
    models.append(model_lightgbm)
    # print(f'Flod{fold}: AUC:{roc_auc_score(y_va, oof_preds[test_index]):.4f}')

# print('CV平均AUC:',roc_auc_score(y_train,oof_preds))
# print(models)
#确定最佳迭代次数
best_iters = [m.best_iteration_ for m in models]
final_n = int(np.mean(best_iters))

#寻找最优阈值
best_thr,best_f1 = 0.5,0
for thr in np.arange(0.01, 1, 0.01):
    f1 = f1_score(y_train, (oof_preds > thr).astype(int))
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

final_model = lgb.LGBMClassifier(
        n_estimators=final_n,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        bagging_seed=42,
        feature_fraction_seed=42,
        verbose=-1
    )

final_model.fit(X_train, y_train)
pred_proba_lightgbm = final_model.predict_proba(X_test)[:,1]


#保存模型
import joblib,os,json

# 创建 models 目录（如果不存在）
os.makedirs('models', exist_ok=True)

# 保存模型
joblib.dump(final_model, 'models/lgbm_model.pkl')

#保存最优阈值
os.makedirs('best_thr', exist_ok=True)
joblib.dump(best_thr, 'models/best_thr.pkl')

#保存特征
with open('models/name_feature.json', 'w',encoding='utf-8') as f:
    json.dump(list(X_train.columns), f, ensure_ascii=False)
#print('模型已保存到 models/lgbm_model.pkl')

print('LightGBM_AUC:',roc_auc_score(y_test,pred_proba_lightgbm))
print('LightGBM_PR-AUC:',average_precision_score(y_test,pred_proba_lightgbm))
print('LightGBM_F1:',f1_score(y_test,(pred_proba_lightgbm>best_thr).astype(int)))

#查看特征重要程度
importances = final_model.feature_importances_
feat_imp = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
print(feat_imp)

import shap

import matplotlib.pyplot as plt

explainer  = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test[:100])
if isinstance(shap_values,list):
    sv = shap_values[1]
else:
    sv = shap_values
#条形图
shap.summary_plot(sv,X_test[:100],plot_type='bar')
plt.show()
#瀑布图
shap.waterfall_plot(
    shap.Explanation(values = sv[0],
    base_values = explainer.expected_value,
    data = X_test.iloc[0],
    feature_names = X_test.columns)
)
plt.show()

y_pred = (pred_proba_lightgbm > best_thr).astype(int)
fp = (y_pred == 1) & (y_test == 0)   # 好人被误判成坏人
fn = (y_pred == 0) & (y_test == 1)   # 坏人被放过

print('好人被误判数量:', fp.sum())
print('坏人被放过数量:', fn.sum())

for col in ['age', 'MonthlyIncome', 'TotalPastDue']:
    print(col, '好人被误判平均:', X_test.loc[fp, col].mean(),
          '坏人被放过平均:', X_test.loc[fn, col].mean())