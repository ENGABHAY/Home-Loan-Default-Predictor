import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from catboost import CatBoostClassifier

# ====================== FEATURE ENGINEERING FUNCTIONS ======================

def engineer_application_features(df):
    df = df.copy()
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).round(1)
    df['AGE_GROUP'] = pd.cut(df['AGE_YEARS'], bins=[0, 25, 35, 45, 55, 65, 100],
                             labels=['Young', 'Thirties', 'Forties', 'Fifties', 'Sixties', 'Senior'])
    df['FLAG_UNEMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365.25).round(1)
    df['EMPLOYED_INCOME_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AMT_INCOME_TOTAL'] / 100000 + 1)
    df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
    doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
    df['NUM_DOCUMENTS_PROVIDED'] = df[doc_cols].sum(axis=1)
    inquiry_cols = [c for c in df.columns if c.startswith('AMT_REQ_CREDIT_BUREAU_')]
    if inquiry_cols:
        df['TOTAL_CREDIT_INQUIRIES'] = df[inquiry_cols].sum(axis=1)
    ext_sources = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    ext_available = [c for c in ext_sources if c in df.columns]
    df['EXT_SOURCE_MEAN'] = df[ext_available].mean(axis=1)
    df['EXT_SOURCE_PROD'] = df[ext_available].prod(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_available].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_available].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_available].max(axis=1)
    df['EXT_SOURCE_COUNT'] = df[ext_available].notna().sum(axis=1)
    df['LOAN_TERM_MONTHS'] = df['AMT_CREDIT'] / (df['AMT_ANNUITY'] + 1)
    df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['EXTRA_BORROWED'] = df['AMT_CREDIT'] - df['AMT_GOODS_PRICE']
    df['EXTRA_BORROWED_RATIO'] = df['EXTRA_BORROWED'] / (df['AMT_GOODS_PRICE'] + 1)
    df['OWNS_CAR_AND_REALTY'] = ((df['FLAG_OWN_CAR'] == 'Y') & (df['FLAG_OWN_REALTY'] == 'Y')).astype(int)
    df['OWNS_NOTHING'] = ((df['FLAG_OWN_CAR'] == 'N') & (df['FLAG_OWN_REALTY'] == 'N')).astype(int)
    return df


def engineer_bureau_balance_features(bureau_balance_df):
    df = bureau_balance_df.copy()
    df['STATUS_C'] = (df['STATUS'] == 'C').astype(int)
    df['STATUS_X'] = (df['STATUS'] == 'X').astype(int)
    df['STATUS_0'] = (df['STATUS'] == '0').astype(int)
    df['STATUS_1'] = (df['STATUS'] == '1').astype(int)
    df['STATUS_2'] = (df['STATUS'] == '2').astype(int)
    df['STATUS_3'] = (df['STATUS'] == '3').astype(int)
    df['STATUS_4'] = (df['STATUS'] == '4').astype(int)
    df['STATUS_5'] = (df['STATUS'] == '5').astype(int)
    df['STATUS_OVERDUE'] = df[['STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5']].sum(axis=1)
    df['STATUS_SERIOUS'] = df[['STATUS_2','STATUS_3','STATUS_4','STATUS_5']].sum(axis=1)
    df['STATUS_VERY_SERIOUS'] = df[['STATUS_3','STATUS_4','STATUS_5']].sum(axis=1)

    agg_dict = {
        'MONTHS_BALANCE': ['count', 'min'],
        'STATUS_C': ['max', 'mean'],
        'STATUS_0': ['mean'],
        'STATUS_OVERDUE': ['sum', 'mean', 'max'],
        'STATUS_SERIOUS': ['sum', 'mean'],
        'STATUS_VERY_SERIOUS': ['sum', 'max'],
        'STATUS_1': ['sum'], 'STATUS_2': ['sum'], 'STATUS_3': ['sum'],
        'STATUS_4': ['sum'], 'STATUS_5': ['sum'],
    }
    bureau_bal_agg = df.groupby('SK_ID_BUREAU').agg(agg_dict)
    bureau_bal_agg.columns = ['BB_' + col[0] + '_' + col[1].upper() for col in bureau_bal_agg.columns]
    bureau_bal_agg = bureau_bal_agg.reset_index()
    total_months = bureau_bal_agg['BB_MONTHS_BALANCE_COUNT']
    bureau_bal_agg['BB_OVERDUE_RATIO'] = bureau_bal_agg['BB_STATUS_OVERDUE_SUM'] / (total_months + 1)
    bureau_bal_agg['BB_SERIOUS_RATIO'] = bureau_bal_agg['BB_STATUS_SERIOUS_SUM'] / (total_months + 1)
    return bureau_bal_agg


def engineer_bureau_features(bureau_df, bureau_bal_agg_df):
    df = bureau_df.copy()
    df = df.merge(bureau_bal_agg_df, on='SK_ID_BUREAU', how='left')
    df['BUREAU_CREDIT_ACTIVE'] = (df['CREDIT_ACTIVE'] == 'Active').astype(int)
    df['BUREAU_CREDIT_CLOSED'] = (df['CREDIT_ACTIVE'] == 'Closed').astype(int)
    df['BUREAU_BAD_DEBT'] = (df['CREDIT_ACTIVE'] == 'Bad debt').astype(int)
    df['BUREAU_OVERDUE'] = (df['CREDIT_DAY_OVERDUE'] > 0).astype(int)
    df['BUREAU_DEBT_RATIO'] = df['AMT_CREDIT_SUM_DEBT'] / (df['AMT_CREDIT_SUM'] + 1)
    df['BUREAU_CREDIT_AGE_YEARS'] = (-df['DAYS_CREDIT'] / 365.25)

    numeric_agg = {
        'SK_ID_BUREAU': ['count'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean', 'sum'],
        'BUREAU_OVERDUE': ['sum', 'mean'],
        'AMT_CREDIT_SUM': ['mean', 'max', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'max', 'sum'],
        'BUREAU_CREDIT_ACTIVE': ['sum', 'mean'],
        'BUREAU_CREDIT_CLOSED': ['sum', 'mean'],
        'BUREAU_DEBT_RATIO': ['mean', 'max'],
        'BUREAU_CREDIT_AGE_YEARS': ['mean', 'max', 'min'],
        'BB_OVERDUE_RATIO': ['mean', 'max'],
        'BB_SERIOUS_RATIO': ['mean', 'max'],
    }
    bureau_agg = df.groupby('SK_ID_CURR').agg(numeric_agg)
    bureau_agg.columns = ['BURO_' + col[0] + '_' + col[1].upper() for col in bureau_agg.columns]
    bureau_agg = bureau_agg.reset_index()
    if 'BURO_SK_ID_BUREAU_COUNT' in bureau_agg.columns:
        bureau_agg.rename(columns={'BURO_SK_ID_BUREAU_COUNT': 'BURO_NUM_CREDITS'}, inplace=True)
    return bureau_agg


def engineer_previous_app_features(prev_df):
    df = prev_df.copy()
    df['PREV_APPROVED'] = (df['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    df['PREV_REFUSED'] = (df['NAME_CONTRACT_STATUS'] == 'Refused').astype(int)
    df['PREV_APP_CREDIT_DIFF'] = df['AMT_APPLICATION'] - df['AMT_CREDIT']
    df['PREV_APP_CREDIT_RATIO'] = df['AMT_APPLICATION'] / (df['AMT_CREDIT'] + 1)
    df['PREV_DAYS_DECISION_ABS'] = abs(df['DAYS_DECISION'])

    agg_dict = {
        'SK_ID_PREV': ['count'],
        'PREV_APPROVED': ['sum', 'mean'],
        'PREV_REFUSED': ['sum', 'mean'],
        'AMT_APPLICATION': ['mean', 'max', 'sum'],
        'AMT_CREDIT': ['mean', 'max', 'sum'],
        'PREV_APP_CREDIT_DIFF': ['mean', 'max'],
        'PREV_APP_CREDIT_RATIO': ['mean'],
        'PREV_DAYS_DECISION_ABS': ['mean', 'min'],
    }
    prev_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    prev_agg.columns = ['PREV_' + col[0] + '_' + col[1].upper() for col in prev_agg.columns]
    prev_agg = prev_agg.reset_index()
    if 'PREV_SK_ID_PREV_COUNT' in prev_agg.columns:
        prev_agg.rename(columns={'PREV_SK_ID_PREV_COUNT': 'PREV_NUM_APPLICATIONS'}, inplace=True)
    return prev_agg


def engineer_pos_cash_features(pos_df):
    df = pos_df.copy()
    df['POS_DPD_FLAG'] = (df['SK_DPD'] > 0).astype(int)
    df['POS_COMPLETED'] = (df['NAME_CONTRACT_STATUS'] == 'Completed').astype(int)

    agg_dict = {
        'SK_ID_PREV': ['count', 'nunique'],
        'SK_DPD': ['max', 'mean'],
        'POS_DPD_FLAG': ['sum', 'mean'],
        'POS_COMPLETED': ['sum', 'mean'],
    }
    pos_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    pos_agg.columns = ['POS_' + col[0] + '_' + col[1].upper() for col in pos_agg.columns]
    pos_agg = pos_agg.reset_index()
    return pos_agg


def engineer_credit_card_features(cc_df):
    df = cc_df.copy()
    df['CC_UTILIZATION'] = df['AMT_BALANCE'] / (df['AMT_CREDIT_LIMIT_ACTUAL'] + 1)

    agg_dict = {
        'SK_ID_PREV': ['nunique'],
        'AMT_BALANCE': ['mean', 'max'],
        'CC_UTILIZATION': ['mean', 'max'],
        'SK_DPD': ['max', 'mean'],
    }
    cc_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    cc_agg.columns = ['CC_' + col[0] + '_' + col[1].upper() for col in cc_agg.columns]
    cc_agg = cc_agg.reset_index()
    return cc_agg


def engineer_installments_features(inst_df):
    df = inst_df.copy()
    df['PAYMENT_DELAY'] = df['DAYS_ENTRY_PAYMENT'] - df['DAYS_INSTALMENT']
    df['LATE_PAYMENT'] = (df['PAYMENT_DELAY'] > 0).astype(int)

    agg_dict = {
        'SK_ID_PREV': ['count'],
        'PAYMENT_DELAY': ['mean', 'max'],
        'LATE_PAYMENT': ['sum', 'mean'],
        'AMT_PAYMENT': ['sum', 'mean'],
        'AMT_INSTALMENT': ['sum', 'mean'],
    }
    inst_agg = df.groupby('SK_ID_CURR').agg(agg_dict)
    inst_agg.columns = ['INST_' + col[0] + '_' + col[1].upper() for col in inst_agg.columns]
    inst_agg = inst_agg.reset_index()
    return inst_agg


def build_final_dataset(app_df, bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg):
    df = app_df.copy()
    df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
    df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
    df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
    df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
    df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
    return df


# =========================== MAIN TRAINING ===========================

if __name__ == "__main__":
    data_dir = "data"
    
    print("Loading datasets...")
    application_train = pd.read_csv(f"{data_dir}/application_train.csv")
    bureau = pd.read_csv(f"{data_dir}/bureau.csv")
    bureau_balance = pd.read_csv(f"{data_dir}/bureau_balance.csv")
    previous_application = pd.read_csv(f"{data_dir}/previous_application.csv")
    credit_card_balance = pd.read_csv(f"{data_dir}/credit_card_balance.csv")
    installments_payments = pd.read_csv(f"{data_dir}/installments_payments.csv")
    POS_CASH_balance = pd.read_csv(f"{data_dir}/POS_CASH_balance.csv")

    print("Splitting data...")
    x = application_train.drop(columns='TARGET')
    y = application_train['TARGET']
    app_train, app_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

    print("Engineering features...")
    app_train = engineer_application_features(app_train)

    bureau_bal_agg = engineer_bureau_balance_features(bureau_balance)
    bureau_agg = engineer_bureau_features(bureau, bureau_bal_agg)
    prev_agg = engineer_previous_app_features(previous_application)
    pos_agg = engineer_pos_cash_features(POS_CASH_balance)
    cc_agg = engineer_credit_card_features(credit_card_balance)
    inst_agg = engineer_installments_features(installments_payments)

    print("Building final dataset...")
    final_df = build_final_dataset(app_train, bureau_agg, prev_agg, pos_agg, cc_agg, inst_agg)
    final_df = final_df.replace([np.inf, -np.inf], np.nan)

    print("Creating preprocessor...")
    cat_cols = list(final_df.select_dtypes(include=['object', 'category']).columns)
    num_cols = [c for c in final_df.columns if c not in ['SK_ID_CURR', 'TARGET'] and 
                final_df[c].dtype != 'object' and not pd.api.types.is_categorical_dtype(final_df[c])]

    numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value=0))])
    categorical_pipeline = Pipeline([('target_encoder', TargetEncoder())])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, num_cols),
        ('cat', categorical_pipeline, cat_cols)
    ])

    X_for_fit = final_df.reset_index(drop=True)
    y_for_fit = y_train.reset_index(drop=True)
    X_processed = preprocessor.fit_transform(X_for_fit, y_for_fit)

    print("Saving preprocessor and feature order...")
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    with open("artifacts/feature_order.pkl", "wb") as f:
        pickle.dump(num_cols + cat_cols, f)

    print("Training CatBoost...")
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = neg / pos

    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=5,
        loss_function='Logloss',
        eval_metric='AUC',
        l2_leaf_reg=15,
        random_strength=2,
        bagging_temperature=1,
        class_weights=[1, class_weight],
        random_seed=42,
        od_type='Iter',
        od_wait=200,
        verbose=100
    )
    model.fit(X_processed, y_train, eval_set=(X_processed, y_train), use_best_model=True)
    model.save_model("artifacts/model.cbm")

    metadata = {"cat_cols": cat_cols, "num_cols": num_cols, "threshold": 0.5}
    with open("artifacts/metadata.json", "w") as f:
        json.dump(metadata, f)

    print("✅ Training completed successfully!")
    print("Artifacts saved in artifacts/ folder.")