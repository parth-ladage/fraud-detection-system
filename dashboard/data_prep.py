import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def main():
    print("Loading data...")
    # Load model to get features
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    features = model_data['features']
    
    # Read the full dataset (might take some time, but it's a one-off script)
    df_trans = pd.read_csv('../data/train_transaction.csv')
    df_id = pd.read_csv('../data/train_identity.csv')
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    print(f"Original shape: {df.shape}")
    
    # Sample data to keep it manageable for Streamlit Cloud
    fraud_df = df[df['isFraud'] == 1]
    legit_df = df[df['isFraud'] == 0].sample(n=min(len(fraud_df)*2, 30000), random_state=42)
    df_sample = pd.concat([fraud_df, legit_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Sampled shape: {df_sample.shape}")
    
    # Preprocessing
    # Separate features and target
    X = df_sample.drop(columns=['isFraud', 'TransactionID'])
    y = df_sample['isFraud']
    
    # We will not dynamically drop columns based on >50% missing in the sample,
    # because the original model already decided which 199 features to keep.
    
    # Identify column types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Impute numerical - median
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
        
    # Impute categorical - mode, then label encode
    le = LabelEncoder()
    for col in cat_cols:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        X[col] = X[col].fillna(mode_val)
        X[col] = le.fit_transform(X[col].astype(str))
        
    # Feature Engineering
    mean_amt = X['TransactionAmt'].mean()
    X['AmtToMeanRatio'] = X['TransactionAmt'] / (mean_amt + 1e-9)
    X['HourOfDay'] = (X['TransactionDT'] // 3600) % 24
    X['DayOfWeek'] = (X['TransactionDT'] // (3600 * 24)) % 7
    X['LogTransactionAmt'] = np.log1p(X['TransactionAmt'])
    
    # Ensure all features needed by the model are present
    missing_features = [f for f in features if f not in X.columns]
    if missing_features:
        print(f"Warning: The following model features are missing and will be imputed with 0: {missing_features}")
        for f in missing_features:
            X[f] = 0
            
    # Filter to only the required columns + identifiers
    cols_to_keep = ['TransactionID'] + features
    final_df = pd.DataFrame()
    final_df['TransactionID'] = df_sample['TransactionID']
    final_df['isFraud'] = df_sample['isFraud']
    
    for f in features:
        final_df[f] = X[f]
        
    # Save as parquet
    out_path = 'sample_data.parquet'
    final_df.to_parquet(out_path, index=False)
    print(f"Saved {len(final_df)} records to {out_path} (Size: {os.path.getsize(out_path)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    main()
