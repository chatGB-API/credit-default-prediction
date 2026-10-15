import streamlit as st
import joblib,json
import numpy as np
import pandas as pd

model = joblib.load('models/lgbm_model.pkl')#
#取出特征文件
with open('models/name_feature.json', 'r',encoding='utf-8') as f:
    feature_names = json.load(f)

st.title('信用违约预测')
st.write(f'模型需要{len(feature_names)}个特征')

#设置参数
st.subheader('客户信息')

revolving = st.slider('信用卡利用率', 0.0, 1.0, 0.3)
age = st.number_input('年龄', 18, 100, 35)
past_due_30_59 = st.slider('30-59天逾期次数', 0, 20, 0)
DebtRatio = st.slider('负债比率',0,5,0)
income = st.number_input('月收入', 0, 50000, 5000)
NumberOfOpenCreditLinesAndLoans = st.number_input('开放信贷帐户数',0,100,1)
NumberOfTimes90DaysLate = st.slider('90天以上逾期次数', 0, 20, 0)
NumberRealEstateLoansOrLines = st.number_input('房产贷款笔数',0,100,0)
NumberOfTime60_89DaysPastDueNotWorse = st.slider('60-89天逾期次数', 0, 20, 0)
NumberOfDependents = st.number_input('瞻养人数',0,100,0)



if st.button('预测'):
    total_past_due = past_due_30_59+NumberOfTime60_89DaysPastDueNotWorse+NumberOfTimes90DaysLate
    income_per_person = income/(NumberOfDependents+1)
    has_real_estate = int(NumberRealEstateLoansOrLines>0)
    late_ratio = total_past_due/(NumberOfOpenCreditLinesAndLoans+1)

    values= {"RevolvingUtilizationOfUnsecuredLines":revolving,
             "age":age,
             "NumberOfTime30_59DaysPastDueNotWorse":past_due_30_59,
             "DebtRatio":DebtRatio,
             "MonthlyIncome":income,
             "NumberOfOpenCreditLinesAndLoans":NumberOfOpenCreditLinesAndLoans,
             "NumberOfTimes90DaysLate":NumberOfTimes90DaysLate,
             "NumberRealEstateLoansOrLines":NumberRealEstateLoansOrLines,
             "NumberOfTime60_89DaysPastDueNotWorse":NumberOfTime60_89DaysPastDueNotWorse,
             "NumberOfDependents":NumberOfDependents,
             "TotalPastDue":total_past_due,
             "IncomePerPerson":income_per_person,
             "HasRealEstate":has_real_estate,
             "LatePaymentRatio":late_ratio}


    X = pd.DataFrame([[values[n] for n in feature_names]] , columns= feature_names)# 按训练时特征顺序
    proba = model.predict_proba(X)[0, 1]
    if proba > 0.5:
        st.error(f'违约概率: {proba:.2%} — 高风险')
    elif proba > 0.2:
        st.warning(f'违约概率: {proba:.2%} — 中风险')
    else:
        st.success(f'违约概率: {proba:.2%} — 低风险')