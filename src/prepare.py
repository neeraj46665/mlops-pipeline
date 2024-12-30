import pandas as pd
import os
import yaml

def load_params():
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def prepare_data():
    params = load_params()
    input_path = params['prepare']['data_path']
    output_path = params['prepare']['output_path']
    
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed data
    df.to_csv(output_path, index=False)
    print(f"Data prepared and saved to '{output_path}'")

if __name__ == "__main__":
    prepare_data()
