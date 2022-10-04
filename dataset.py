from torch.utils.data import Dataset, DataLoader
import pandas as pd

class dataset(Dataset):
    def __init__(self, csv_path):
        self.labels=[]
        self.texts=[]
        data=pd.read_csv(csv_path)
        for i in range(data.shape[0]):
            self.labels.append(data['label'][i])
            self.texts.append(data['text'][i])

    def __getitem__(self, index):
        label = self.labels[index]
        text = self.texts[index]
        return text, label

    def __len__(self):
        return len(self.texts)

    
