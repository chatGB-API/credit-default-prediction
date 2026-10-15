from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# 建议用相对路径，或者确保路径正确
model = joblib.load(
    'D:/study/compute/jiqixuexi/credit-default-prediction/models/lgbm_model.pkl'
)

# 可选：启动时打印模型期望的特征数量，方便核对
# print("model n_features_in_:", getattr(model, "n_features_in_", None))
# print("model feature_name_:", getattr(model, "feature_name_", None))


class Customer(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: float
    past_due_30_59: float
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float
    NumberOfDependents: float


@app.post('/predict')
def predict(c: Customer):
    # 1. 取出原始特征
    revolving = c.RevolvingUtilizationOfUnsecuredLines
    age = c.age
    past_due_30_59 = c.past_due_30_59
    debt_ratio = c.DebtRatio
    monthly_income = c.MonthlyIncome
    open_credit_lines = c.NumberOfOpenCreditLinesAndLoans
    times_90_late = c.NumberOfTimes90DaysLate
    real_estate_loans = c.NumberRealEstateLoansOrLines
    past_due_60_89 = c.NumberOfTime60_89DaysPastDueNotWorse
    dependents = c.NumberOfDependents

    # 2. 计算衍生特征
    total_past_due = past_due_30_59 + past_due_60_89 + times_90_late
    income_per_person = monthly_income / (dependents + 1)
    has_real_estate = int(real_estate_loans > 0)
    late_ratio = total_past_due / (open_credit_lines + 1)

    # 3. 按训练模型时的特征顺序拼接
    #    这里假设模型用了 10 个原始特征 + 4 个衍生特征 = 14 个
    #    实际顺序必须与训练时完全一致！
    X = np.array([[
        revolving,
        age,
        past_due_30_59,
        debt_ratio,
        monthly_income,
        open_credit_lines,
        times_90_late,
        real_estate_loans,
        past_due_60_89,
        dependents,
        total_past_due,
        income_per_person,
        has_real_estate,
        late_ratio,
    ]])

    # 4. 预测
    proba = model.predict_proba(X)[0, 1]
    return {
        'probability': float(proba),
        'risk': 'high' if proba > 0.5 else 'low'
    }