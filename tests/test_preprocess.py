"""preprocess.py 的单元测试。运行方式：pytest tests/ -v"""
from pathlib import Path
import pandas as pd
import pytest

from src.preprocess import load_and_clean, engineer_features, clean_column_names

DATA_PATH = 'data/raw/cs-training.csv'


@pytest.fixture(scope='module')
def raw_df():
    """读取一次，多个测试共用。"""
    return load_and_clean(DATA_PATH)


def test_no_missing_income(raw_df):
    """清洗后 MonthlyIncome 不应再有缺失。"""
    assert raw_df['MonthlyIncome'].isnull().sum() == 0


def test_no_missing_dependents(raw_df):
    """清洗后 NumberOfDependents 不应再有缺失。"""
    assert raw_df['NumberOfDependents'].isnull().sum() == 0


def test_revolving_clipped(raw_df):
    """信用卡利用率应被截断到 [0, 1]。"""
    col = raw_df['RevolvingUtilizationOfUnsecuredLines']
    assert col.max() <= 1
    assert col.min() >= 0


def test_age_filtered(raw_df):
    """年龄应全部大于 18。"""
    assert (raw_df['age'] > 18).all()


def test_engineer_features_created(raw_df):
    """特征工程应生成 4 个新列。"""
    df = engineer_features(raw_df)
    for col in ['TotalPastDue', 'IncomePerPerson', 'HasRealEstate', 'LatePaymentRatio']:
        assert col in df.columns, f'缺少特征: {col}'


def test_total_past_due_correct(raw_df):
    """TotalPastDue 应等于三个逾期次数之和。"""
    df = engineer_features(raw_df)
    expected = (
        df['NumberOfTime30-59DaysPastDueNotWorse']
        + df['NumberOfTime60-89DaysPastDueNotWorse']
        + df['NumberOfTimes90DaysLate']
    )
    assert (df['TotalPastDue'] == expected).all()


def test_clean_column_names():
    """列名中的非法字符应被替换成下划线。"""
    df = pd.DataFrame({'a-b': [1], 'c d': [2], 'e.f': [3]})
    df = clean_column_names(df)
    assert list(df.columns) == ['a_b', 'c_d', 'e_f']