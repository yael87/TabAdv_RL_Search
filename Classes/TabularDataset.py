from torch.utils.data import Dataset
import torch
import pandas as pd

class TabularDataset(Dataset):
    """tabular dataset."""

    def __init__(self, x_data, y_data=None, transform=None):
        """
        Arguments:
            x_data (pd.DataFrame):
            y_data (DataFrame):
            transform (callable, optional): Optional transform (scaler) to be applied
                on a sample.
        """
        self.x_data = x_data
        self.y_data = y_data
        self.transform = transform

        if self.transform:
            self.x_data = pd.DataFrame(self.transform.transform(x_data), columns= x_data.columns)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x_data = torch.tensor(self.x_data.iloc[idx, 0:]).float()
        if self.y_data is not None:
            y_data = torch.tensor(self.y_data.iloc[idx, 0:]).float()
        else:
            y_data = torch.tensor([])
        
        sample = {'x_data': x_data, 'y_data': y_data}

        return sample