import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torcheval.metrics.functional import mean_squared_error
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

pd.set_option('future.no_silent_downcasting', True)
#torch.manual_seed(42)
#torch.use_deterministic_algorithms(True)


class mlp_model(torch.nn.Module):
    def __init__(self, num_hidden, input_size, hidden_size, output_size, dropout):
        
        super().__init__()
        self.model = torch.nn.ModuleList()

        self.model.append(torch.nn.Linear(input_size, hidden_size))
        self.model.append(torch.nn.ReLU())
        for layer in range(num_hidden):
            self.model.append(torch.nn.Linear(hidden_size, hidden_size))
            self.model.append(torch.nn.ReLU())
            self.model.append(torch.nn.Dropout(dropout))
        self.model.append(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x):
        for layer in self.model:
            #print(x.shape)
            x = layer(x)
        return x

class Experiment:
    def __init__(self, config_path, data_path, target_col, response_col):
    
        self.config = pd.read_csv(config_path)
        self.response_col = response_col
        #print(self.config.head(10))
    
        self.data = pd.read_csv(data_path)
        self.target_col = target_col

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print("Experiment Initialized on device", self.device)

    def describe_data(self):      
        print('\n\nNUMERICAL FEATURES')
        print(self.data.describe())

    def clean_data(self):

        view = self.data.infer_objects(copy=False)

        mms = MinMaxScaler()
        ss = StandardScaler()

        for col in view.columns:
            transformer = ss
            if col != 'quality':
                view[col] = transformer.fit_transform(view[[col]])
                view[col] = view[col].astype('float64')
        
        return view

    def train_model(self, arg_model, learning_rate, train_loader, test_loader, epochs):
        opt = Adam(arg_model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()
        
        arg_model.train()
        for epoch in range(epochs):
            for x, y in train_loader:
                # take a batch
                # forward pass
                y_pred = arg_model(x).squeeze(-1).to(self.device)
                #print(y_pred.shape, ybatch.shape)
                #print(y_pred, ybatch)
                loss = loss_fn(y_pred, y)
                # backward pass
                opt.zero_grad(set_to_none=True)
                loss.backward()
                # update weights
                opt.step()
            if(epoch%100==0):
                print(str(epoch) + '/' + str(epochs))
        
        # evaluate trained model with test set
        arg_model.eval()
        mse = []
        for x_val, y_val in test_loader:
            with torch.no_grad():
                y_pred = arg_model(x_val).squeeze(-1)
            #print(y_val, y_pred.T, predicted_labels.T)
            mse.append(y_pred - y_val)
        score = float(torch.mean(torch.tensor([mse])))
        print("Mean of Residuals: {:.2f}".format(score))
        return score

    def run_trial(self, config):
        
        print("Trial run", config.name, ":", int(config['Pattern']))

        model = mlp_model(
            int(config['Hidden Layers']/2),
            input_size=len(self.data.columns)-1,
            hidden_size=64,
            output_size=1,
            dropout=config['Dropout Ratio']
        ).to(self.device)
        
        features = self.data.drop([self.target_col], axis='columns')
        targets = self.data[self.target_col]

        x_train, x_test, y_train, y_test = train_test_split(
            torch.tensor(features.values, dtype=torch.float32).to(self.device),
            torch.tensor(targets.values, dtype=torch.float32).to(self.device),
            #random_state=69,
            train_size=0.75,
            shuffle=True
        )

        train_loader = DataLoader(list(zip(x_train, y_train)), shuffle=True, batch_size=64)
        test_loader = DataLoader(list(zip(x_test, y_test)), shuffle=False)

        #print(self.config.iloc[[config.name]])
        self.config.loc[config.name, 'System Performance'] = self.train_model(model, config['Learning Rate'], train_loader, test_loader, 100)


if __name__ == '__main__':
    exp = Experiment('config.csv','winequality-red.csv', 'quality', 'System Performance')
    exp.describe_data()
    exp.data = exp.clean_data()
    exp.describe_data()

    result_df = exp.config.apply(exp.run_trial, axis='columns')

    exp.config.to_csv('wine_post_hoc.csv')