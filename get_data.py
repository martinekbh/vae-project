from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

class COIL20(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.root_dir = root_dir
        self.labels = range(1,21)
        self.objects = ["Rubber duck",
            "Wooden toy 1",
            "Toy car 1",
            "Lucky cat",
            "Anacin",
            "Toy car 2",
            "Wooden toy 2",
            "Johnsons baby powder",
            "Tylenol",
            "Vaseline",
            "Wooden toy 3",
            "Chinese cup",
            "Piggy bank",
            "Connector",
            "Plastic container",
            "Conditioner bottle",
            "Ceramic pot",
            "Teacup",
            "Toy car 3",
            "Philadelphia"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_id, img_label = row['image'], row['label']
        img_filename = self.root_dir + '/' + str(img_id)
        img = Image.open(img_filename)
        if self.transform:
            img = self.transform(img)
        return img, img_label