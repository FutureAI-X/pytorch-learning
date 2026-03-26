from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from pathlib import Path

# 获取项目根路径
current_file = Path(__file__).resolve()
root_path = current_file.parent.parent.parent

def get_training_data():
    return datasets.FashionMNIST(
        root=f"{root_path}/data_handle/datasets",
        train=True,
        download=True,
        transform=ToTensor()
    )

def get_training_dataloader(batch_size:int = 64):
    return DataLoader(get_training_data(), batch_size=batch_size)

def get_test_data():
    return datasets.FashionMNIST(
        root=f"{root_path}/data_handle/datasets",
        train=False,
        download=True,
        transform=ToTensor()
    )

def get_test_dataloader(batch_size:int = 64):
    return DataLoader(get_test_data(), batch_size=batch_size)