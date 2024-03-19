import pandas as pd
import torch



def Experiment(config_path, data_path, target_col, response_col):
    
    #create empty config csv if one is not provided.
    #config = pd.DataFrame(columns=[
    #    'Pattern',
    #    'Hidden Layers',
    #    'Learning Rate',
    #    'Dropout Ratio',
    #    'System Performance',
    #])
    #config.to_csv('config.csv')

    config = pd.read_csv(config_path)
    print(config.head(10))
    
    heart_data = pd.read_csv(data_path)
    print(heart_data.head(10))



def describe_data(data):
    pass

def clean_data(heart_data):
    pass

def generate_model(layer_size=32):
    pass

if __name__ == '__main__':
    Experiment('config.csv','heart.csv', 'HeartDisease', 'System Performance')