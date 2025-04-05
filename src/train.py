import torch
from torch.utils.data import DataLoader
from data import DataClass
from typing import Literal
from


class Runner:
    def __init__(
            self,
            TrainDataLoader: DataLoader,
            ValDataLoader: DataLoader,
            TestDataLoader: DataLoader,
            Model: torch.nn.Module,
            Loss: torch.nn.Module,
            Optimizer: torch.optim.Optimizer
    ):
        self.TrainDataLoader = TrainDataLoader
        self.ValDataLoader = ValDataLoader
        self.TestDataLoader = TestDataLoader
        self.Model = Model
        self.Loss = Loss
        self.Optimizer = Optimizer
        pass

    def train(self):
        for dTrainData in self.TrainDataLoader:
            dTrainData: dict
            tPred = self.Model(dTrainData['tImage'])
            tLabel = dTrainData['iLabel']
            loss = self.Loss(tPred, tLabel)
            self.Optimizer.zero_grad()
            loss.backward()
            self.Optimizer.step()

    @torch.no_grad()
    def val(self):
        for dTrainData in self.ValDataLoader:
            dTrainData: dict
            tPred = self.Model(dTrainData['tImage'])
            tLabel = dTrainData['iLabel']
            loss = self.Loss(tPred, tLabel)

    @torch.no_grad()
    def test(self):
        for dTrainData in self.TestDataLoader:
            dTrainData: dict
            tPred = self.Model(dTrainData['tImage'])
            tLabel = dTrainData['iLabel']
            loss = self.Loss(tPred, tLabel)

    def run(self, sMode: Literal['train', 'infer'] = 'infer', iEpochLen=100):
        for iEpoch in range(iEpochLen):
            if sMode == 'train':
                self.train()
                self.val()

            elif sMode == 'infer':
                self.test()


if __name__ == '__main__':
    _sTrainPath: str = rf'G:\data\test\label'
    _sValPath: str = rf'G:\data\test\label'
    _sTestPath: str = rf'G:\data\test\label'
    _sOutPath: str = rf'G:\data\test\model'
    _iEpochLen: int = 100
    _sMode: Literal['train', 'infer']  = 'train'


    _TrainDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sTrainPath),
        batch_size=32,
        shuffle=True
    )
    _ValDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sValPath),
        batch_size=32,
        shuffle=True
    )
    _TestDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sTestPath),
        batch_size=32,
        shuffle=True
    )
    _Model = model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    runner = Runner(
        TrainDataLoader=_TrainDataLoader,
        ValDataLoader=_ValDataLoader,
        TestDataLoader=_TestDataLoader,
        Model=_Model,
        Loss=torch.nn.CrossEntropyLoss(),
        Optimizer=torch.optim.Adam(_Model.parameters(), lr=1e-3)
    )

    runner.run(sMode=_sMode, iEpochLen=_iEpochLen)



