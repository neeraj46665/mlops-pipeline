import pandas as pd
import os
import yaml
from sklearn.preprocessing import StandardScaler

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def featureize_data():
    params = load_params()
    input_path = params['featureize']['input_path']
    output_path = params['featureize']['output_path']
    
    # Load the processed data
    df = pd.read_csv(input_path)
    
    # Separate features and target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=df.columns[:-1])
    df_scaled['target'] = y
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the featureized data
    df_scaled.to_csv(output_path, index=False)
    print(f"Data featureized and saved to '{output_path}'")

if __name__ == "__main__":
    featureize_data()
