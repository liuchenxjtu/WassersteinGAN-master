import torch.utils.data
import pandas as pd
import numpy as np
class DatasetFromPandas(torch.utils.data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromPandas, self).__init__()
        df = pd.read_csv(file_path)
        self.data = df[df.columns[1:-1]]

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.data.iloc[index])).float()

    def __len__(self):
        return self.data.shape[0]
