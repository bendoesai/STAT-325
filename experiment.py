import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam

pd.set_option('future.no_silent_downcasting', True)
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)


class Experiment:
    def __init__(self, config_path, data_path, target_col, response_col):
    
        self.config = pd.read_csv(config_path)
        self.response_col = response_col
        #print(self.config.head(10))
    
        self.heart_data = pd.read_csv(data_path)
        self.categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        self.numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
        self.target_col = target_col

        self.device = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        print("Experiment Initialized on device", self.device)

    def describe_data(self):
        
        print('\n\nCATEGORICAL FEATURES')
        for col in self.categorical_cols:
            print(self.heart_data[col].unique())

        
        print('\n\nNUMERICAL FEATURES')
        print(self.heart_data[self.numerical_cols].describe())

    def clean_data(self):
        categorical_map = {
            'Sex': {
                'M': -1,
                'F': 1,
            },
            'ChestPainType': {
                'ATA': -3,
                'NAP': -1,
                'ASY': 1,
                'TA' : 3,
            },
            'RestingECG':{
                'ST':       -1,
                'Normal':   0,
                'LVH':      1,
            },
            'ExerciseAngina':{
                'Y': 1,
                'N': 0,
            },
            'ST_Slope':{
                'Up':   1,
                'Flat': 0,
                'Down': -1,
            },
        }

        view = self.heart_data.infer_objects(copy=False)

        for col in self.categorical_cols:
            view[col] = view[col].replace(categorical_map[col])
            view[col] = view[col].astype('int64')
        
        return view

    def train_model(self, arg_model, learning_rate, x_train, y_train, x_val, y_val, epochs):
        opt = Adam(arg_model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        batch_size = 16  # size of each batch
        batches_per_epoch = len(x_train) // batch_size
        arg_model.train()
        for epoch in range(epochs):
            for i in range(batches_per_epoch):
                start = i * batch_size
                # take a batch
                Xbatch = x_train[start:start+batch_size].to(self.device)
                ybatch = y_train[start:start+batch_size].to(self.device)
                # forward pass
                y_pred = arg_model(Xbatch).squeeze(-1)
                #print(y_pred.shape, ybatch.shape)
                #print(y_pred, ybatch)
                loss = loss_fn(y_pred, ybatch)
                # backward pass
                opt.zero_grad()
                loss.backward()
                # update weights
                opt.step()
            if(epoch%10==0):
                print(str(epoch) + '/' + str(epochs))
        
        # evaluate trained model with test set
        arg_model.eval()
        with torch.no_grad():
            y_pred = arg_model(x_val)
        predicted_labels = y_pred > 0.0
        print(y_val, y_pred.T, predicted_labels.T)
        accuracy = torch.mean((y_val == predicted_labels).float())
        print("Accuracy {:.2f}".format(accuracy * 100))
        return float(accuracy*100)

    def run_trial(self, config):
        
        print("Trial run:", config['Pattern'])

        model = mlp_model(int(config['Hidden Layers']),
                          input_size=len(self.heart_data.columns)-1,
                          hidden_size=32,
                          output_size=1,
                          dropout=config['Dropout Ratio']
                        ).to(self.device)
        
        features = self.heart_data.drop([self.target_col], axis='columns')
        targets = self.heart_data[self.target_col]

        x_train, x_test, y_train, y_test = train_test_split(
            torch.tensor(features.values, dtype=torch.float32),
            torch.tensor(targets.values, dtype=torch.float32),
            random_state=69,
            train_size=0.75
        )

        #print(self.config.iloc[[config.name]])
        self.config.loc[config.name, 'System Performance'] = self.train_model(model, config['Learning Rate'], x_train, y_train, x_test, y_test, 500)
    
    def run():
        pass

class mlp_model(torch.nn.Module):
    def __init__(self, num_hidden, input_size, hidden_size, output_size, dropout):
        
        super().__init__()
        self.model = torch.nn.ModuleList()

        self.model.append(torch.nn.Linear(input_size, hidden_size))
        self.model.append(torch.nn.LeakyReLU())
        for layer in range(num_hidden):
            self.model.append(torch.nn.Linear(hidden_size, hidden_size))
            self.model.append(torch.nn.Dropout(dropout))
            self.model.append(torch.nn.LeakyReLU())
        self.model.append(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.model:
            #print(x.shape)
            x = layer(x)
        return x


if __name__ == '__main__':
    exp = Experiment('config.csv','heart.csv', 'HeartDisease', 'System Performance')
    #exp.describe_data()
    exp.heart_data = exp.clean_data()

    result_df = exp.config.apply(exp.run_trial, axis='columns')

    exp.config.to_csv('post_hoc.csv')