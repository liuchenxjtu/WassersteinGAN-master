import torch.utils.data
import pandas as pd
import numpy as np
class DatasetFromPandas(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromPandas, self).__init__()
        df = pd.read_csv(file_path)
        sub_data = df[df['label'].isin(['NG','OK'])]
        self.data = sub_data[sub_data.columns[1:-1]]
        self.label = sub_data['label'].apply(lambda x:1 if x=='OK' else 0)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data.iloc[index]))

    def __len__(self):
        return self.data.shape[0]
