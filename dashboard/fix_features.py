import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def get_exact_features():
    print("Loading full dataset to get exact 199 features...")
    df_trans = pd.read_csv('../data/train_transaction.csv')
    df_id = pd.read_csv('../data/train_identity.csv')
    df = pd.merge(df_trans, df_id, on='TransactionID', how='left')
    
    # Preprocessing
    X = df.drop(columns=['isFraud', 'TransactionID'])
    y = df['isFraud'].astype(int)
    
    # Drop columns with > 50% missing values
    missing_pct = X.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    X = X.drop(columns=cols_to_drop)
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
        
    le = LabelEncoder()
    for col in cat_cols:
        mode_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
        X[col] = X[col].fillna(mode_val)
        X[col] = le.fit_transform(X[col].astype(str))
        
    mean_amt = X['TransactionAmt'].mean()
    X['AmtToMeanRatio'] = X['TransactionAmt'] / (mean_amt + 1e-9)
    X['HourOfDay'] = (X['TransactionDT'] // 3600) % 24
    X['DayOfWeek'] = (X['TransactionDT'] // (3600 * 24)) % 7
    X['LogTransactionAmt'] = np.log1p(X['TransactionAmt'])
    
    # Stratified 80/20 split
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Variance filtering
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(X_train)
    selected_feature_names = X.columns[selector.get_support()].tolist()
    
    print(f"Number of selected features: {len(selected_feature_names)}")
    
    # Fix the model.pkl to have the correct features
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        
    model_data['features'] = selected_feature_names
    
    with open('model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("Fixed model.pkl with correct 199 features.")
    
    with open('correct_features.pkl', 'wb') as f:
        pickle.dump(selected_feature_names, f)

if __name__ == "__main__":
    get_exact_features()
