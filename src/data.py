import os
import os.path as osp
import json
from dataclasses import dataclass
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as T


class ClassColor:
    t0: tuple = (255, 0, 0)
    t1: tuple = (0, 255, 0)
    t2: tuple = (0, 0, 255)
    t3: tuple = (255, 255, 0)
    t4: tuple = (255, 0, 255)


@dataclass
class Label:
    sFileName: str
    iLabel: int
    iX: int
    iY: int
    iW: int
    iH: int
    tColor: tuple
    sEncodingTest: str


@dataclass
class Data:
    tImage: str
    iLabel: int
    iX: int
    iY: int
    iW: int
    iH: int
    tColor: tuple


def load_json(
        path: str,
        encoding: str = 'cp949'
) -> dict:
    with open (path, 'r', encoding=encoding) as f:
        dJsonDict = json.load(f)

    return dJsonDict


class DataClass(Dataset):
    def __init__(self, sJsonBasePath: str):
        self.m_lItemList = []
        sJsonBasePath: str = sJsonBasePath
        lJsonList = os.listdir(sJsonBasePath)

        for sJson in lJsonList:
            dItemDict = load_json(osp.join(sJsonBasePath, sJson))
            self.m_lItemList.append(dItemDict)

    def __len__(self):
        return len(self.m_lItemList)

    def __getitem__(self, key):
        dSelectItem = self.m_lItemList[key]
        nImage: np.ndarray = np.array(Image.open(dSelectItem['sFilePath']).convert('RGB'))
        tImage: Tensor = self.transform(nImage)

        dResultDict = {}
        dResultDict['tImage'] = tImage
        dResultDict['nImage'] = nImage
        dResultDict['iLabel'] = dSelectItem['lDataList']

        return

    def transform(self, nImage: np.ndarray) -> Tensor:
        lTransformList = [
            T.ToTensor()
        ]

        return torch.Compose(lTransformList)(nImage)





if __name__ == '__main__':
    _sInPath = rf'G:\data\test\label'

    _DataClass = DataClass(sJsonBasePath=_sInPath)
    a = 0

