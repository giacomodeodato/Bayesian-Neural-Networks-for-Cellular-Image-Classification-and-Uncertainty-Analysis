import torch
from pybbbc import BBBC021 as Dataset

class BBBC021(torch.utils.data.Dataset):
    """Creates the BBBC021 dataset compatible with pytorch DataLoader."""
    
    def __init__(self, path: str = './data/bbbc021.h5', **kwargs) -> None:
        """Initializes the BBBC021 Dataset.
        
        This dataset is based on the BBBC021 dataset class implemented 
        in [1] and the data available from the Broad Bioimage Benchmark 
        Collection website [2].

        Parameters
        ----------
        path : str, optional
            Path to the HDF5 virtual dataset file.
        kwargs : dict()
            Keyword arguments to filter the dataset based on metadata.

        References
        ----------
            [1] https://github.com/giacomodeodato/pybbbc

            [2] https://bbbc.broadinstitute.org/BBBC021
        """

        super(BBBC021, self).__init__()
        self.dataset = Dataset(path, **kwargs)
    
    def __getitem__(self, index: int) -> (torch.Tensor, torch.Tensor, tuple):
        """Returns the sample at the specified index.

        A sample is a tuple (image, target, metadata) made of the cellular
        image, the corresponding Mechanism of Action (target) and the associated
        metadata in the following format:

        metadata = (
            ( # plate metadata
                site,
                well,
                replicate,
                plate
            ), 
            ( # compound metadata
                compound,
                concentration,
                moa
            )
        )

        Parameters
        ----------
        index : int
            Index of the sample to retrieve.

        Returns
        -------
        (torch.Tensor, torch.Tensor, tuple)
            Dataset sample made of image, target and metadata.
        """
        
        img, metadata = self.dataset[index]
        img = torch.from_numpy(img)

        _, (_, _, moa) = metadata
        moa = Dataset.MOA.index(moa)
        moa = torch.tensor(moa)

        return img, moa, metadata
    
    def __len__(self) -> int:
        """Returns the length of the dataset."""

        return len(self.dataset)