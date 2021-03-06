import torch
from pybbbc import BBBC021 as Dataset

class BBBC021(torch.utils.data.Dataset):
    
    def __init__(self, path='./data/bbbc021.h5', **kwargs) -> None:
        super(BBBC021, self).__init__()
        self.dataset = Dataset(path, **kwargs)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, tuple):
        
        img, metadata = self.dataset[index]
        img = torch.from_numpy(img)

        _, (_, _, moa) = metadata
        moa = Dataset.MOA.index(moa)
        moa = torch.tensor(moa)

        return img, moa, metadata
    
    def __len__(self) -> int:
        return len(self.dataset)