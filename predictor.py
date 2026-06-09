import pandas as pd
import numpy as np
import pickle
import json
import os
import warnings
warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier

class LoanDefaultPredictor:
    def __init__(self, artifacts_path="artifacts"):
        self.artifacts_path = artifacts_path
        self.model = None
        self.preprocessor = None
        self.feature_order = None
        self.metadata = None
        self.load_artifacts()

    def load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        print("Loading model artifacts...")
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(self.artifacts_path, "model.cbm"))
        
        with open(os.path.join(self.artifacts_path, "preprocessor.pkl"), "rb") as f:
            self.preprocessor = pickle.load(f)
            
        with open(os.path.join(self.artifacts_path, "feature_order.pkl"), "rb") as f:
            self.feature_order = pickle.load(f)
            
        with open(os.path.join(self.artifacts_path, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
        print("✅ All artifacts loaded successfully!")

    # ====================== FEATURE ENGINEERING (Exact from training) ======================
    def engineer_application_features(self, df):
        df = df.copy()
        df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / (df['AMT_GOODS_PRICE'] + 1)
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
        df['AGE_YEARS'] = (-df['DAYS_BIRTH'] / 365.25).round(1)
        df['FLAG_UNEMPLOYED'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
        df['EMPLOYMENT_YEARS'] = (-df['DAYS_EMPLOYED'] / 365.25).round(1)
        df['EMPLOYED_INCOME_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AMT_INCOME_TOTAL'] / 100000 + 1)
        df['EMPLOYMENT_TO_AGE_RATIO'] = df['EMPLOYMENT_YEARS'] / (df['AGE_YEARS'] + 1)
        
        doc_cols = [c for c in df.columns if c.startswith('FLAG_DOCUMENT_')]
        if doc_cols:
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
        
        df['OWNS_CAR_AND_REALTY'] = ((df.get('FLAG_OWN_CAR', 'N') == 'Y') & 
                                     (df.get('FLAG_OWN_REALTY', 'N') == 'Y')).astype(int)
        df['OWNS_NOTHING'] = ((df.get('FLAG_OWN_CAR', 'N') == 'N') & 
                              (df.get('FLAG_OWN_REALTY', 'N') == 'N')).astype(int)
        return df

    # Add other engineer_* functions if needed (simplified version used in training)
    # For now we use the ones from training

    def prepare_input(self, application_df, **kwargs):
        """Fixed version - prevents row explosion during merge"""
        print("Engineering features...")
        app_eng = self.engineer_application_features(application_df.copy())
        df = app_eng.copy()
        
        print("Merging auxiliary tables (with aggregation to avoid row duplication)...")
        
        for key, table in kwargs.items():
            if table is not None and not table.empty:
                # Critical fix: Aggregate auxiliary tables before merging
                if 'SK_ID_CURR' in table.columns:
                    # Simple aggregation to keep one row per customer
                    table_agg = table.groupby('SK_ID_CURR').mean(numeric_only=True).reset_index()
                    df = df.merge(table_agg, on='SK_ID_CURR', how='left')
                else:
                    df = df.merge(table, on='SK_ID_CURR', how='left')

        df = df.replace([np.inf, -np.inf], np.nan)

        # Align columns
        for col in self.feature_order:
            if col not in df.columns:
                df[col] = np.nan

        print(f"Final shape before preprocessor: {df.shape}")
        
        print("Applying preprocessor...")
        X_processed = self.preprocessor.transform(df)
        
        return pd.DataFrame(X_processed, columns=self.feature_order, index=application_df.index)    
    
    def predict_proba(self, application_df, **kwargs):
        """Return default probability"""
        X = self.prepare_input(application_df, **kwargs)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, application_df, threshold=0.5, **kwargs):
        """Return binary prediction"""
        proba = self.predict_proba(application_df, **kwargs)
        return (proba >= threshold).astype(int)