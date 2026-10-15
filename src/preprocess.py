import pandas as pd
import re
import numpy as np

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    #数据处理
    #用中位数填充收入缺失
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    #家属人数缺失填零
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    # 信用卡利用率超过 1 的截断到 1
    df['RevolvingUtilizationOfUnsecuredLines'] = df['RevolvingUtilizationOfUnsecuredLines'].clip(0,1)
    # 年龄小于 18 的删掉
    df = df[df['age'] > 18]
    return df

def engineer_features(df):

    #特征工程
    #逾期总次数
    df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse']
                          + df['NumberOfTime60-89DaysPastDueNotWorse']
                          + df['NumberOfTimes90DaysLate'])
    #人均收入
    df['IncomePerPerson'] = (
        df['MonthlyIncome']/(df['NumberOfDependents']+1)
    )
    # 是否有房产贷款
    df['HasRealEstate'] = (df['NumberRealEstateLoansOrLines'] > 0).astype(int)
    # 逾期比例
    df['LatePaymentRatio'] = df['TotalPastDue'] / (df['NumberOfOpenCreditLinesAndLoans'] + 1)
    return df

def clean_column_names(df):

    # 清洗列名
    df.columns = [
        re.sub(r'_+', '_', re.sub(r'[^0-9a-zA-Z_]', '_', str(c).strip())).strip('_') or f'f{i}'
        for i, c in enumerate(df.columns)
        ]
    return df