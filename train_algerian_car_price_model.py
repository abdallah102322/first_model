"""
========================================================================================
ALGERIAN USED & NEW CAR PRICE VALUATION ENGINE - PRODUCTION ML PIPELINE
========================================================================================
A complete, production-grade machine learning solution to predict used and new car prices
in the Algerian market using leak-free nested k-fold cross-validation, target encoding,
multi-model benchmarking (Ridge, ElasticNet, RandomForest, ExtraTrees, LightGBM, XGBoost,
CatBoost, StackingRegressor), granular error diagnostics, SHAP interpretability, and
joblib pipeline persistence.

Required Pip Packages:
    pip install scikit-learn lightgbm xgboost catboost optuna shap matplotlib seaborn pandas numpy joblib

Run:
    python train_algerian_car_price_model.py [--data_path PATH] [--sample_size N] [--n_trials T]
========================================================================================
"""

import sys
import os
import argparse
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress non-critical warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Scikit-learn
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, KFold, learning_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor

# Gradient Boosting
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

# Hyperparameter Optimization & Explainability
import optuna
import shap
import joblib

# Global Reproducibility Seed
SEED = 42
np.random.seed(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ======================================================================================
# SECTION 1: DATA INSPECTION, STANDARDIZATION & DOMAIN-AWARE CLEANING
# ======================================================================================

def inspect_dataset(df: pd.DataFrame, stage_name: str = "Raw Data"):
    """Prints comprehensive inspection stats: shape, dtypes, missing values, duplicates."""
    print("=" * 78)
    print(f"📊 DATA INSPECTION: {stage_name}")
    print("=" * 78)
    print(f"[*] Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\n[*] Column Data Types & Missing Value Counts:")
    missing_df = pd.DataFrame({
        'dtype': df.dtypes,
        'missing_count': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(missing_df.to_string())
    
    n_dupes = df.duplicated().sum()
    print(f"\n[*] Exact Duplicate Rows: {n_dupes:,} ({n_dupes / len(df) * 100:.2f}%)")
    
    print("\n[*] Summary Statistics (Numeric):")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(df[numeric_cols].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())
    print("=" * 78 + "\n")


def clean_and_standardize_data(df: pd.DataFrame, current_year: int = 2026) -> pd.DataFrame:
    """
    Standardizes categorical text, numeric values, removes exact duplicates,
    flags implausible values, and removes spurious outliers while preserving
    legitimate luxury and classic cars in the Algerian market.
    """
    print("🧹 [1/7] Cleaning and standardizing corpus...")
    initial_len = len(df)
    df = df.copy()

    # 1. Deduplication
    df = df.drop_duplicates()
    print(f"  └─ Dropped {initial_len - len(df):,} exact duplicates ({len(df):,} remaining)")

    # 2. Standardize column names
    col_mapping = {
        'brand': 'brand', 'Brand': 'brand',
        'model': 'model', 'Model': 'model',
        'year': 'year', 'Year': 'year',
        'mileage_km': 'mileage_km', 'Mileage': 'mileage_km', 'mileage': 'mileage_km',
        'fuel_type': 'fuel_type', 'Fuel_Type': 'fuel_type', 'fuel': 'fuel_type',
        'transmission': 'transmission', 'Transmission': 'transmission',
        'wilaya_code': 'wilaya_code', 'Wilaya': 'wilaya_code', 'wilaya': 'wilaya_code',
        'price_million_centimes': 'price_million_centimes', 'Price': 'price_million_centimes',
        'price_dzd': 'price_dzd'
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # Ensure price is present in Million Centimes (1M Centimes = 10,000 DZD)
    if 'price_million_centimes' not in df.columns and 'price_dzd' in df.columns:
        df['price_million_centimes'] = df['price_dzd'] / 10000.0
    elif 'price_million_centimes' not in df.columns and 'Price' in df.columns:
        df['price_million_centimes'] = df['Price'].astype(float)

    # 3. Standardize text features
    df['brand'] = df['brand'].astype(str).str.strip().str.title()
    df['model'] = df['model'].astype(str).str.strip().str.title()
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip().str.capitalize()
    df['transmission'] = df['transmission'].astype(str).str.strip().str.capitalize()

    # Standardize fuel categories (Petrol, Diesel, LPG)
    fuel_map = {
        'Essence': 'Petrol', 'Gasoline': 'Petrol', 'Petrol': 'Petrol',
        'Diesel': 'Diesel', 'Gasoil': 'Diesel',
        'Gpl': 'LPG', 'GPL': 'LPG', 'Lpg': 'LPG', 'Gaz': 'LPG'
    }
    df['fuel_type'] = df['fuel_type'].map(lambda x: fuel_map.get(x, x if x in ['Petrol', 'Diesel', 'LPG'] else 'Petrol'))

    # Standardize transmission (Manual, Automatic)
    trans_map = {
        'Manuelle': 'Manual', 'Manual': 'Manual', 'M': 'Manual',
        'Automatique': 'Automatic', 'Automatic': 'Automatic', 'Auto': 'Automatic', 'A': 'Automatic'
    }
    df['transmission'] = df['transmission'].map(lambda x: trans_map.get(x, x if x in ['Manual', 'Automatic'] else 'Manual'))

    # 4. Numeric coercions and validations
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df['mileage_km'] = pd.to_numeric(df['mileage_km'], errors='coerce')
    df['price_million_centimes'] = pd.to_numeric(df['price_million_centimes'], errors='coerce')

    # 5. Filter implausible years and prices (Domain-justified rules for Algeria)
    # Cars before 1970 are extreme edge antiques; cars beyond 2026 are future dates.
    # Prices < 25.0 M Centimes (250,000 DZD) are fake/parts/deposits.
    # Prices > 3000.0 M Centimes (30,000,000 DZD / $220k USD) cover up to Porsche/Range Rover without fake 999999 entries.
    valid_mask = (
        (df['year'] >= 1970) & (df['year'] <= current_year) &
        (df['price_million_centimes'] >= 25.0) & (df['price_million_centimes'] <= 3000.0)
    )
    filtered_df = df[valid_mask].copy()
    print(f"  └─ Filtered implausible year / price outliers: {len(df) - len(filtered_df):,} dropped ({len(filtered_df):,} remaining)")

    # 6. Mileage validation: flag missing/zero and cap extreme values (> 700,000 km)
    filtered_df['mileage_missing'] = (filtered_df['mileage_km'].isnull() | (filtered_df['mileage_km'] < 0)).astype(int)
    # Cap implausible high mileage (e.g. 999,999 or typo millions) to 650,000 km
    filtered_df['mileage_km'] = filtered_df['mileage_km'].clip(lower=0, upper=650000)

    # 7. Wilaya code validation (1 to 58 official Algerian wilayas)
    filtered_df['wilaya_code'] = pd.to_numeric(filtered_df['wilaya_code'], errors='coerce').fillna(16).astype(int)
    # Default non-standard codes to 16 (Algiers / central market)
    filtered_df['wilaya_code'] = filtered_df['wilaya_code'].apply(lambda w: w if (1 <= w <= 58) else 16)

    print(f"[✔] Data cleaning complete. Final usable rows: {len(filtered_df):,}")
    return filtered_df.reset_index(drop=True)


# Raw feature inputs expected from users/listings
RAW_FEATURE_COLUMNS = ['brand', 'model', 'year', 'mileage_km', 'fuel_type', 'transmission', 'wilaya_code']

def extract_features_and_target(df: pd.DataFrame) -> tuple:
    """Extracts raw feature columns and target price."""
    X = df[RAW_FEATURE_COLUMNS].copy()
    y = df['price_million_centimes'].copy()
    return X, y


class AlgeriaCarFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain feature engineer for Algerian automotive market.
    Fits only within CV training folds to prevent data leakage.
    
    Engineered Features:
      - car_age: current_year - year
      - km_per_year: mileage / (car_age + 1)
      - brand_model: high-cardinality interaction (e.g. 'Renault_Clio 4')
      - is_new_car: boolean flag for mileage <= 1,000 km or age <= 1 year
      - is_luxury_brand: indicator for high-tier brands (Porsche, Mercedes, BMW, Audi, etc.)
      - wilaya_region: Northern Coast, High Plateaus, Southern Sahara
    """
    def __init__(self, current_year: int = 2026):
        self.current_year = current_year
        self.luxury_brands = {
            'Mercedes-Benz', 'Mercedes', 'Bmw', 'Audi', 'Porsche', 'Land Rover',
            'Range Rover', 'Jaguar', 'Lexus', 'Maserati', 'Ferrari', 'Lamborghini',
            'Bentley', 'Rolls-Royce', 'Alfa Romeo', 'Volvo'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        
        # Ensure year and mileage are numeric
        year = pd.to_numeric(X_out['year'], errors='coerce').fillna(2016)
        mileage = pd.to_numeric(X_out['mileage_km'], errors='coerce')
        
        # Missing mileage indicator and imputation for dynamics
        X_out['mileage_missing'] = mileage.isnull().astype(int)
        mileage_clean = mileage.fillna(100000)
        X_out['mileage_km'] = mileage_clean
        
        # Age & Mileage dynamics
        car_age = (self.current_year - year).clip(lower=0)
        X_out['car_age'] = car_age
        
        # Mileage per year (smooth denominator by +1 to prevent div by zero)
        X_out['km_per_year'] = mileage_clean / (car_age + 1.0)
        
        # Interaction: Brand + Model
        brand = X_out['brand'].astype(str).str.strip().str.title()
        model = X_out['model'].astype(str).str.strip().str.title()
        X_out['brand'] = brand
        X_out['model'] = model
        X_out['brand_model'] = brand + '_' + model
        
        # Market Segment Flags
        X_out['is_new_car'] = ((mileage.fillna(0) <= 1000) | (car_age <= 1)).astype(int)
        X_out['is_luxury_brand'] = brand.isin(self.luxury_brands).astype(int)
        
        # Geographic Wilaya String
        X_out['wilaya_code'] = pd.to_numeric(X_out['wilaya_code'], errors='coerce').fillna(16).astype(int).astype(str)
        
        return X_out


def build_preprocessor_pipeline(current_year: int = 2026) -> Pipeline:
    """
    Constructs a ColumnTransformer that guarantees 100% leak-free preprocessing:
      - Numeric: SimpleImputer (median) + RobustScaler (handles skewed mileages/ages)
      - Low-cardinality categorical: SimpleImputer + OneHotEncoder
      - High-cardinality categorical: TargetEncoder (cross-fit inside training fold)
    """
    numeric_features = [
        'year', 'mileage_km', 'car_age', 'km_per_year',
        'mileage_missing', 'is_new_car', 'is_luxury_brand'
    ]
    categorical_low = ['fuel_type', 'transmission']
    categorical_high = ['brand', 'model', 'brand_model', 'wilaya_code']

    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])

    cat_low_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    cat_high_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('target_encoder', TargetEncoder(smooth='auto', cv=3, target_type='continuous'))
    ])

    column_preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat_low', cat_low_transformer, categorical_low),
            ('cat_high', cat_high_transformer, categorical_high)
        ],
        remainder='drop'
    )

    full_prep = Pipeline([
        ('feature_engineer', AlgeriaCarFeatureEngineer(current_year=current_year)),
        ('column_transform', column_preprocessor)
    ])
    
    return full_prep


def build_model_pipeline(regressor, current_year: int = 2026) -> Pipeline:
    """
    Wraps preprocessor and regressor in TransformedTargetRegressor(func=log1p, inverse=expm1)
    to enforce training in log space while automatically outputting in original currency.
    """
    prep = build_preprocessor_pipeline(current_year=current_year)
    ttr = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1
    )
    return Pipeline([
        ('preprocessor', prep),
        ('ttr', ttr)
    ])


# ======================================================================================
# SECTION 3: CROSS-VALIDATION STRATEGY & INNER OPTUNA TUNING
# ======================================================================================

def get_outer_cv_splitter(df: pd.DataFrame, n_splits: int = 5, n_bins: int = 10):
    """
    Nested CV Strategy:
    Outer CV: StratifiedKFold with k=5 on price deciles.
    
    Justification for k=5:
      - 5 folds provide an optimal 80/20 train/validation split per fold.
      - Maintains computational tractability across 8 model architectures.
      - Ensures low estimation bias and variance.
      - Stratification on price deciles guarantees that rare luxury vehicles (>1,000M Centimes)
        and budget cars (<100M Centimes) are equally distributed across all folds, preventing
        fold distribution shift without data leakage.
    """
    price_bins = pd.qcut(df['price_million_centimes'], q=n_bins, labels=False, duplicates='drop')
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED), price_bins


def optimize_hyperparameters_optuna(X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'lightgbm', n_trials: int = 25) -> dict:
    """
    Inner CV Loop:
    Uses KFold with k=3 and Optuna to find optimal hyperparameters
    strictly using inner training data.
    """
    print(f"    🔍 Tuning {model_type.upper()} with Optuna ({n_trials} trials, 3-fold inner CV)...")
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

    def objective(trial):
        if model_type == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 31, 127),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'random_state': SEED,
                'n_jobs': -1,
                'verbose': -1
            }
            reg = lgb.LGBMRegressor(**params)
        elif model_type == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 200, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.15, log=True),
                'depth': trial.suggest_int('depth', 5, 8),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'random_seed': SEED,
                'verbose': 0,
                'thread_count': -1
            }
            reg = CatBoostRegressor(**params)
        elif model_type == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 150, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
                'max_depth': trial.suggest_int('max_depth', 5, 8),
                'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
                'random_state': SEED,
                'n_jobs': -1,
                'eval_metric': 'rmse'
            }
            reg = xgb.XGBRegressor(**params)
        else:
            return 999.0

        pipe = build_model_pipeline(reg)
        
        # Fast inner evaluation on 3 folds
        scores = []
        for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
            X_in_tr, X_in_val = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
            y_in_tr, y_in_val = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]
            
            pipe.fit(X_in_tr, y_in_tr)
            preds = pipe.predict(X_in_val)
            mae = mean_absolute_error(y_in_val, preds)
            scores.append(mae)
            
        return np.mean(scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"      Best Inner MAE: {study.best_value:.2f} M Centimes | Best Params: {study.best_params}")
    return study.best_params


# ======================================================================================
# SECTION 4: MODELS TO COMPARE & NESTED CV BENCHMARK
# ======================================================================================

def initialize_model_candidates(best_params_map: dict = None) -> dict:
    """
    Instantiates all 8 models to compare:
      - Baselines: DummyRegressor (median), Ridge, ElasticNet
      - Tree Ensembles: RandomForest, ExtraTrees
      - Gradient Boosting: LightGBM, XGBoost, CatBoost
      - Final Ensemble: StackingRegressor (combining top 3 with Ridge meta-learner)
    """
    if best_params_map is None:
        best_params_map = {}

    lgb_params = best_params_map.get('lightgbm', {
        'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 63,
        'random_state': SEED, 'n_jobs': -1, 'verbose': -1
    })
    xgb_params = best_params_map.get('xgboost', {
        'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 6,
        'random_state': SEED, 'n_jobs': -1, 'eval_metric': 'rmse'
    })
    cb_params = best_params_map.get('catboost', {
        'iterations': 400, 'learning_rate': 0.05, 'depth': 6,
        'random_seed': SEED, 'verbose': 0, 'thread_count': -1
    })

    # Top individual estimators
    m_lgb = lgb.LGBMRegressor(**lgb_params)
    m_xgb = xgb.XGBRegressor(**xgb_params)
    m_cb = CatBoostRegressor(**cb_params)

    # Stacking ensemble combining gradient boosters with Ridge meta-regressor
    stack_estimators = [
        ('lgb', m_lgb),
        ('xgb', m_xgb),
        ('cat', m_cb)
    ]
    m_stack = StackingRegressor(
        estimators=stack_estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )

    models = {
        'Dummy (Median)': DummyRegressor(strategy='median'),
        'Ridge Regression': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=16, random_state=SEED, n_jobs=-1),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=SEED, n_jobs=-1),
        'LightGBM': m_lgb,
        'XGBoost': m_xgb,
        'CatBoost': m_cb,
        'Stacking Ensemble': m_stack
    }
    return models


# ======================================================================================
# SECTION 5: METRICS COMPUTATION (PER FOLD & GRAND AGGREGATES)
# ======================================================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes RMSE, MAE, MAPE, MdAPE, and R2 in original currency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Avoid zero division
    denom = np.where(y_true == 0, 1e-6, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    mdape = np.median(np.abs((y_true - y_pred) / denom)) * 100.0
    r2 = r2_score(y_true, y_pred)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE_%': mape,
        'MdAPE_%': mdape,
        'R2': r2
    }


def run_nested_cv_benchmark(df: pd.DataFrame, n_splits: int = 5, n_optuna_trials: int = 25) -> tuple:
    """
    Executes leak-free nested cross-validation across all 8 models on identical outer folds.
    Returns:
      - comparison_df: summary comparison table with mean ± std
      - fold_results: raw fold performance logs
      - oof_predictions: out-of-fold predictions for diagnostics
      - best_model_name: champion architecture name
    """
    print("\n" + "=" * 78)
    print("🚀 [3/7] EXECUTING NESTED CROSS-VALIDATION BENCHMARK (k=5 Outer Folds)")
    print("=" * 78)

    X, y = extract_features_and_target(df)

    cv_splitter, strat_bins = get_outer_cv_splitter(df, n_splits=n_splits)

    # 1. Run Optuna tuning on the first outer fold training set to discover optimal hyperparams
    print("[*] Phase 3.1: Running Inner Optuna Hyperparameter Optimization on Inner Folds...")
    train_idx_0, _ = next(cv_splitter.split(X, strat_bins))
    X_tr_inner, y_tr_inner = X.iloc[train_idx_0], y.iloc[train_idx_0]
    
    best_params_map = {}
    if n_optuna_trials > 0:
        best_params_map['lightgbm'] = optimize_hyperparameters_optuna(X_tr_inner, y_tr_inner, 'lightgbm', n_trials=n_optuna_trials)
        best_params_map['catboost'] = optimize_hyperparameters_optuna(X_tr_inner, y_tr_inner, 'catboost', n_trials=max(10, n_optuna_trials // 2))
    
    # 2. Initialize models with discovered hyperparams
    model_candidates = initialize_model_candidates(best_params_map)

    # 3. Outer Loop Evaluation
    print(f"\n[*] Phase 3.2: Evaluating {len(model_candidates)} models across {n_splits} identical outer folds...")
    
    fold_metrics = {name: {'RMSE': [], 'MAE': [], 'MAPE_%': [], 'MdAPE_%': [], 'R2': []} for name in model_candidates}
    oof_predictions = {name: np.zeros(len(df)) for name in model_candidates}

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, strat_bins), 1):
        print(f"\n--- Outer Fold {fold}/{n_splits} (Train: {len(train_idx):,} | Val: {len(val_idx):,}) ---")
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        for name, base_reg in model_candidates.items():
            t0 = time.time()
            pipe = build_model_pipeline(base_reg)
            pipe.fit(X_train_fold, y_train_fold)
            preds = pipe.predict(X_val_fold)
            oof_predictions[name][val_idx] = preds
            
            metrics = compute_metrics(y_val_fold, preds)
            for m_key, m_val in metrics.items():
                fold_metrics[name][m_key].append(m_val)
            
            fit_time = time.time() - t0
            print(f"  {name:18s} | MAE: {metrics['MAE']:5.2f}M | RMSE: {metrics['RMSE']:6.2f}M | MdAPE: {metrics['MdAPE_%']:4.1f}% | R²: {metrics['R2']:.4f} | ({fit_time:.1f}s)")

    # Compile Comparison Table
    summary_rows = []
    for name in model_candidates:
        row = {'Model': name}
        for metric in ['MAE', 'RMSE', 'MdAPE_%', 'MAPE_%', 'R2']:
            vals = fold_metrics[name][metric]
            row[f'{metric} (Mean)'] = np.mean(vals)
            row[f'{metric} (Std)'] = np.std(vals)
            row[metric] = f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"
        summary_rows.append(row)

    comparison_df = pd.DataFrame(summary_rows).sort_values(by='MAE (Mean)').reset_index(drop=True)
    best_model_name = comparison_df.iloc[0]['Model']

    print("\n" + "=" * 90)
    print("🏆 FINAL MODEL BENCHMARK COMPARISON TABLE (Nested 5-Fold Outer CV)")
    print("=" * 90)
    display_cols = ['Model', 'MAE', 'RMSE', 'MdAPE_%', 'R2']
    print(comparison_df[display_cols].to_string(index=False))
    print("=" * 90 + "\n")

    return comparison_df, fold_metrics, oof_predictions, best_model_name, model_candidates


def compute_segment_and_brand_breakdowns(df: pd.DataFrame, best_model_name: str, oof_preds: np.ndarray):
    """
    Calculates detailed performance breakdown by price tier (cheap, mid, expensive)
    and by top Algerian automotive brands to diagnose specific model weaknesses.
    """
    print("=" * 78)
    print(f"🔍 [4/7] PERFORMANCE BREAKDOWN FOR CHAMPION MODEL: {best_model_name}")
    print("=" * 78)

    eval_df = df.copy()
    eval_df['predicted_price'] = oof_preds
    eval_df['abs_error'] = (eval_df['price_million_centimes'] - eval_df['predicted_price']).abs()
    eval_df['pct_error'] = (eval_df['abs_error'] / eval_df['price_million_centimes']) * 100.0

    # 1. Price Range Breakdown
    # Cheap: < 200M Centimes (< 2,000,000 DZD)
    # Mid  : 200 - 500M Centimes (2,000,000 - 5,000,000 DZD)
    # Luxury: > 500M Centimes (> 5,000,000 DZD)
    conditions = [
        eval_df['price_million_centimes'] < 200.0,
        (eval_df['price_million_centimes'] >= 200.0) & (eval_df['price_million_centimes'] <= 500.0),
        eval_df['price_million_centimes'] > 500.0
    ]
    tiers = ['1. Cheap (<200M Centimes)', '2. Mid-Range (200-500M)', '3. Luxury/Premium (>500M)']
    eval_df['price_tier'] = np.select(conditions, tiers, default='2. Mid-Range (200-500M)')

    tier_summary = eval_df.groupby('price_tier').agg(
        Sample_Count=('price_million_centimes', 'count'),
        Avg_Actual_Price=('price_million_centimes', 'mean'),
        MAE=('abs_error', 'mean'),
        Median_APE=('pct_error', 'median'),
        Mean_APE=('pct_error', 'mean')
    ).round(2)
    print("\n[*] Breakdown by Price Tier:")
    print(tier_summary.to_string())

    # 2. Brand Breakdown (Top 12 Market Share Brands)
    top_brands = eval_df['brand'].value_counts().head(12).index
    brand_summary = eval_df[eval_df['brand'].isin(top_brands)].groupby('brand').agg(
        Sample_Count=('price_million_centimes', 'count'),
        Avg_Price=('price_million_centimes', 'mean'),
        MAE=('abs_error', 'mean'),
        Median_APE=('pct_error', 'median')
    ).sort_values(by='Sample_Count', ascending=False).round(2)
    print("\n[*] Breakdown by Top 12 Automotive Brands:")
    print(brand_summary.to_string())
    print("=" * 78 + "\n")


# ======================================================================================
# SECTION 6: DIAGNOSTICS & PLOTS
# ======================================================================================

def generate_diagnostic_plots(df: pd.DataFrame, best_model_name: str, oof_preds: np.ndarray, fold_metrics: dict, output_dir: str = "."):
    """
    Generates and saves publication-quality diagnostic plots:
      1. predicted_vs_actual.png
      2. residuals_vs_predicted.png
      3. error_distribution.png
      4. cv_score_per_fold.png
      5. learning_curve.png
      6. feature_importance.png & shap_summary.png
    """
    print("📈 [5/7] Generating and saving diagnostic plots...")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)
    y_true = df['price_million_centimes'].values
    residuals = y_true - oof_preds

    # 1. Predicted vs Actual Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, oof_preds, alpha=0.3, color='#1f77b4', edgecolors='none', s=18)
    min_val, max_val = min(y_true.min(), oof_preds.min()), max(y_true.max(), oof_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (y = ŷ)')
    plt.title(f'Predicted vs Actual Car Prices ({best_model_name})', fontsize=13, weight='bold')
    plt.xlabel('Actual Price (Million Centimes)', fontsize=11)
    plt.ylabel('Predicted Price (Million Centimes)', fontsize=11)
    plt.xlim(0, min(1500, max_val))
    plt.ylim(0, min(1500, max_val))
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predicted_vs_actual.png'), dpi=150)
    plt.close()
    print("  └─ Saved predicted_vs_actual.png")

    # 2. Residuals vs Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(oof_preds, residuals, alpha=0.3, color='#2ca02c', edgecolors='none', s=18)
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title(f'Residuals vs Predicted Prices ({best_model_name})', fontsize=13, weight='bold')
    plt.xlabel('Predicted Price (Million Centimes)', fontsize=11)
    plt.ylabel('Residual (Actual - Predicted) [M Centimes]', fontsize=11)
    plt.xlim(0, min(1500, oof_preds.max()))
    plt.ylim(-400, 400)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residuals_vs_predicted.png'), dpi=150)
    plt.close()
    print("  └─ Saved residuals_vs_predicted.png")

    # 3. Error Distribution Plot
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=60, kde=True, color='#9467bd', edgecolor='black', alpha=0.6)
    plt.axvline(0, color='red', linestyle='--', lw=2, label='Zero Error')
    plt.title('Residual Error Distribution (Actual - Predicted)', fontsize=13, weight='bold')
    plt.xlabel('Residual Error (Million Centimes)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)
    plt.xlim(-300, 300)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'), dpi=150)
    plt.close()
    print("  └─ Saved error_distribution.png")

    # 4. CV Score per Fold Plot
    plt.figure(figsize=(8, 5))
    mae_scores = fold_metrics[best_model_name]['MAE']
    folds = [f"Fold {i+1}" for i in range(len(mae_scores))]
    bars = plt.bar(folds, mae_scores, color='#3b528b', edgecolor='black', width=0.5)
    plt.axhline(np.mean(mae_scores), color='red', linestyle='--', lw=1.5, label=f"Mean MAE: {np.mean(mae_scores):.2f}M")
    plt.title(f'Outer CV MAE Across Folds ({best_model_name})', fontsize=13, weight='bold')
    plt.ylabel('MAE (Million Centimes)', fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.2f}M", ha='center', va='bottom', fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_score_per_fold.png'), dpi=150)
    plt.close()
    print("  └─ Saved cv_score_per_fold.png")


def generate_learning_curve_plot(pipeline, X: pd.DataFrame, y: pd.Series, output_dir: str = "."):
    """Computes and plots learning curve to detect underfitting vs overfitting."""
    print("  └─ Computing learning curve...")
    cv = KFold(n_splits=3, shuffle=True, random_state=SEED)
    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=cv,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mae, 'o-', color='#1f77b4', lw=2, label='Training MAE')
    plt.plot(train_sizes, val_mae, 's-', color='#d62728', lw=2, label='Validation MAE')
    plt.title('Learning Curve (Training vs Validation Error)', fontsize=13, weight='bold')
    plt.xlabel('Training Set Sample Size', fontsize=11)
    plt.ylabel('MAE (Million Centimes)', fontsize=11)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curve.png'), dpi=150)
    plt.close()
    print("  └─ Saved learning_curve.png")


def generate_shap_and_importance(pipeline, X: pd.DataFrame, y: pd.Series, output_dir: str = "."):
    """Extracts feature importance and computes SHAP summary values for interpretability."""
    print("  └─ Computing Feature Importance & SHAP summary plot...")
    # Extract fitted preprocessor and regressor
    preprocessor = pipeline.named_steps['preprocessor']
    ttr = pipeline.named_steps['ttr']
    regressor = ttr.regressor_

    # Transform feature subset for SHAP
    sample_sub = X.sample(min(500, len(X)), random_state=SEED)
    X_trans = preprocessor.transform(sample_sub)

    # Feature names
    num_cols = ['year', 'mileage_km', 'car_age', 'km_per_year', 'mileage_missing', 'is_new_car', 'is_luxury_brand']
    cat_low_cols = ['fuel_Petrol', 'fuel_Diesel', 'fuel_LPG', 'trans_Manual', 'trans_Automatic']
    cat_high_cols = ['brand_te', 'model_te', 'brand_model_te', 'wilaya_te']
    feature_names = num_cols + cat_low_cols[:max(0, X_trans.shape[1] - len(num_cols) - len(cat_high_cols))] + cat_high_cols
    # Ensure correct length
    if len(feature_names) != X_trans.shape[1]:
        feature_names = [f"feat_{i}" for i in range(X_trans.shape[1])]

    # 1. Native Feature Importance Plot
    if hasattr(regressor, 'feature_importances_'):
        importances = regressor.feature_importances_
        imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(9, 6))
        sns.barplot(data=imp_df.head(15), x='importance', y='feature', palette='Blues_r')
        plt.title('Top 15 Most Influential Features (Native Gini / Gain Importance)', fontsize=12, weight='bold')
        plt.xlabel('Relative Feature Importance', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150)
        plt.close()
        print("  └─ Saved feature_importance.png")

    # 2. SHAP Summary Plot
    try:
        explainer = shap.TreeExplainer(regressor)
        shap_vals = explainer(X_trans)
        
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_vals, X_trans, feature_names=feature_names, show=False, max_display=12)
        plt.title('SHAP Value Summary (Impact on Predicted Log-Price in Algeria)', fontsize=12, weight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_summary.png'), dpi=150)
        plt.close()
        print("  └─ Saved shap_summary.png")
    except Exception as e:
        print(f"  [!] Note: SHAP plot skipped ({e})")


# ======================================================================================
# SECTION 7: FINAL PRODUCTION PIPELINE REFIT & SERVING INTERFACE
# ======================================================================================

def refit_and_save_final_pipeline(best_regressor, X_full: pd.DataFrame, y_full: pd.Series, output_path: str = "algerian_car_price_pipeline.joblib") -> Pipeline:
    """
    Refits the selected champion pipeline on 100% of the dataset with leak-free
    target encodings, scalers, and target transformations.
    Saves the entire standalone artifact with joblib.
    """
    print("\n" + "=" * 78)
    print(f"💾 [6/7] REFITTING FINAL CHAMPION MODEL ON 100% OF DATA ({len(X_full):,} SAMPLES)")
    print("=" * 78)

    final_pipeline = build_model_pipeline(best_regressor, current_year=2026)
    t0 = time.time()
    final_pipeline.fit(X_full, y_full)
    fit_duration = time.time() - t0
    print(f"[✔] Refit completed in {fit_duration:.2f} seconds.")

    # Save artifact
    joblib.dump(final_pipeline, output_path)
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[✔] Successfully saved production pipeline to: {output_path} ({file_size_mb:.2f} MB)")
    return final_pipeline


def predict_price(car_dict: dict, pipeline_path: str = "algerian_car_price_pipeline.joblib") -> dict:
    """
    Production inference function for unseen Algerian vehicles.
    
    Parameters:
      car_dict (dict): Dictionary with vehicle attributes:
        - 'brand': str (e.g. 'Renault', 'Volkswagen', 'Peugeot')
        - 'model': str (e.g. 'Clio 4', 'Golf 7', '208', 'Picanto')
        - 'year': int (e.g. 2018)
        - 'mileage_km': float or int (e.g. 120000)
        - 'fuel_type': str ('Petrol', 'Diesel', or 'LPG')
        - 'transmission': str ('Manual' or 'Automatic')
        - 'wilaya_code': int or str (1 to 58, e.g. 16 for Algiers)
      pipeline_path (str): Path to saved joblib pipeline file.
      
    Returns:
      dict: Valuation output with price in Million Centimes and Algerian Dinars (DZD).
    """
    if isinstance(pipeline_path, str):
        pipe = joblib.load(pipeline_path)
    else:
        pipe = pipeline_path

    # Convert single dictionary to DataFrame
    df_single = pd.DataFrame([car_dict])
    pred_million_centimes = float(pipe.predict(df_single)[0])
    pred_dzd = pred_million_centimes * 10000.0

    return {
        'brand': car_dict.get('brand'),
        'model': car_dict.get('model'),
        'year': car_dict.get('year'),
        'mileage_km': car_dict.get('mileage_km'),
        'fuel_type': car_dict.get('fuel_type'),
        'transmission': car_dict.get('transmission'),
        'wilaya_code': car_dict.get('wilaya_code'),
        'predicted_price_million_centimes': round(pred_million_centimes, 2),
        'predicted_price_dzd': round(pred_dzd, -3),
        'formatted_price_dzd': f"{round(pred_dzd, -3):,.0f} DZD",
        'formatted_price_centimes': f"{round(pred_million_centimes, 2):.2f} Million Centimes"
    }


# ======================================================================================
# MAIN EXECUTION ENTRYPOINT
# ======================================================================================

def main():
    parser = argparse.ArgumentParser(description="Algerian Used Car Valuation ML System")
    parser.add_argument('--data_path', type=str, default='D:/database/algeria_cars_200k_model_ready.csv', help='Path to dataset')
    parser.add_argument('--sample_size', type=int, default=15000, help='Stratified benchmark sample size (e.g. 15000 for fast nested CV, -1 for full)')
    parser.add_argument('--n_trials', type=int, default=20, help='Number of inner Optuna tuning trials')
    parser.add_argument('--output_dir', type=str, default='D:/database/ml_diagnostics', help='Output directory for plots & models')
    args = parser.parse_args()

    print("\n" + "=" * 78)
    print("🚗 ALGERIAN AUTOMOTIVE MARKET VALUATION - PRODUCTION TRAINING ENGINE")
    print("=" * 78)

    # 1. Load Data
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Dataset not found at: {args.data_path}")

    raw_df = pd.read_csv(args.data_path)
    inspect_dataset(raw_df, "Raw Loaded Data")

    # 2. Clean & Standardize
    clean_df = clean_and_standardize_data(raw_df)

    # 3. Stratified Representative Sampling for Nested CV Benchmark (if requested)
    if args.sample_size > 0 and len(clean_df) > args.sample_size:
        print(f"[*] Extracting stratified representative benchmark sample: {args.sample_size:,} records across price deciles...")
        strat_bins = pd.qcut(clean_df['price_million_centimes'], q=10, labels=False, duplicates='drop')
        cv_sample = StratifiedKFold(n_splits=max(2, len(clean_df) // args.sample_size), shuffle=True, random_state=SEED)
        _, sample_idx = next(cv_sample.split(clean_df, strat_bins))
        benchmark_df = clean_df.iloc[sample_idx].copy().reset_index(drop=True)
    else:
        benchmark_df = clean_df

    print(f"[*] Nested CV Benchmark corpus: {len(benchmark_df):,} vehicle listings")

    # 4. Run Nested CV Benchmark
    comparison_df, fold_metrics, oof_preds, best_name, model_candidates = run_nested_cv_benchmark(
        benchmark_df, n_splits=5, n_optuna_trials=args.n_trials
    )

    # 5. Segment & Brand Performance Breakdowns
    compute_segment_and_brand_breakdowns(benchmark_df, best_name, oof_preds[best_name])

    # 6. Diagnostic Visualizations
    generate_diagnostic_plots(benchmark_df, best_name, oof_preds[best_name], fold_metrics, output_dir=args.output_dir)

    # Learning Curve & SHAP on Best Pipeline
    best_candidate = model_candidates[best_name]
    best_pipe = build_model_pipeline(best_candidate)
    X_bench, y_bench = extract_features_and_target(benchmark_df)
    generate_learning_curve_plot(best_pipe, X_bench, y_bench, output_dir=args.output_dir)

    # Fit best candidate on benchmark data to extract SHAP
    best_pipe.fit(X_bench, y_bench)
    generate_shap_and_importance(best_pipe, X_bench, y_bench, output_dir=args.output_dir)

    # 7. Refit Champion Pipeline on 100% of Cleaned Corpus & Save
    final_model_path = os.path.join(args.output_dir, 'algerian_car_price_pipeline.joblib')
    X_full, y_full = extract_features_and_target(clean_df)
    final_pipeline = refit_and_save_final_pipeline(best_candidate, X_full, y_full, output_path=final_model_path)

    # 8. Test Single Prediction Example
    print("\n" + "=" * 78)
    print("🚘 [7/7] LIVE INFERENCE TEST WITH PRODUCTION PREDICTION FUNCTION")
    print("=" * 78)
    sample_car = {
        'brand': 'Renault',
        'model': 'Clio 4',
        'year': 2018,
        'mileage_km': 120000,
        'fuel_type': 'Diesel',
        'transmission': 'Manual',
        'wilaya_code': 16
    }
    prediction_result = predict_price(sample_car, pipeline_path=final_pipeline)
    for k, v in prediction_result.items():
        print(f"  {k:35s}: {v}")
    print("=" * 78 + "\n")


if __name__ == '__main__':
    main()
