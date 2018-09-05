import torch.utils.data as data
import pandas as pd

class CocoDataset(data.Dataset):
    """Coco dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        return None


#plt.imshow(image)
#plt.show
caption_lengths =

all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
