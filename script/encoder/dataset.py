from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        split: str,
        transform: Callable,
        **kwargs
    ):
        self.data_dir = Path(data_path) / "office_128x96"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0


class VAEDataset(LightningDataModule):
    """
    Args:
        patch_size: the size of the crop to 
        num_workers: the number of parallel workers to create to load data items
        pin_memory: whether prepared items should be loaded into pinned memory or not. This can improve performance on GPUs
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def setup(self, stage: Optional[str] = None) -> None:

        train_tansforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.patch_size),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        val_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(self.patch_size),
            transforms.Resize(self.patch_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.train_dataset = MyDataset(
            self.data_dir,
            split='train',
            transform=train_tansforms,
        )

        self.val_dataset = MyDataset(
            self.data_dir,
            split='test',
            transform=val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory
        )