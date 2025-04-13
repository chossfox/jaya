import torch
from torch.utils.data import DataLoader
from data import DataClass
from typing import Literal
from tqdm import tqdm
from ioStream import BufferedInputStream


class Runner:
    def __init__(
            self,
            iBatchSize: int,
            TrainDataLoader: DataLoader,
            ValDataLoader: DataLoader,
            TestDataLoader: DataLoader,
            Model: torch.nn.Module,
            Loss: torch.nn.Module,
            Optimizer: torch.optim.Optimizer
    ):
        self.iBatchSize: int = iBatchSize
        self.TrainDataLoader: DataLoader = TrainDataLoader
        self.ValDataLoader: DataLoader = ValDataLoader
        self.TestDataLoader: DataLoader = TestDataLoader
        self.Model: torch.nn.Module = Model
        self.Loss: torch.nn.Module = Loss
        self.Optimizer: torch.optim.Optimizer = Optimizer
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
    def test(self, BufferStream):

        if BufferStream.is_video_open():
            while BufferStream.is_video_open():
                tPred = self.Model(BufferStream.get_frame(frame_num=self.iBatchSize))
                tLabel = torch.full([self.iBatchSize], -1)
        else:
            for dTrainData in self.TestDataLoader:
                dTrainData: dict
                tPred = self.Model(dTrainData['tImage'])
                tLabel = dTrainData['iLabel']
                loss = self.Loss(tPred, tLabel)

    def run(
        self,
        BufferStream: BufferedInputStream,
        sRunMode: Literal['train', 'infer'] = 'infer',
        iEpochLen=100
    ):
        for iEpoch in range(iEpochLen):
            if sRunMode == 'train':
                self.train()
                self.val()

            elif sRunMode == 'infer':
                self.test(BufferStream)


if __name__ == '__main__':
    # Todo: export 만들어야 됨

    _sTrainPath: str = rf'G:\data\test\label'
    _sValPath: str = rf'G:\data\test\label'
    _sTestPath: str = rf'G:\data\test\label'
    _sVideoPath: str = rf'C:\Users\jaya\Videos\NVIDIA\League of Legends\League of Legends 2025.03.17 - 19.14.04.02.mp4'
    _sOutPath: str = rf'G:\data\test\model'
    _iEpochLen: int = 100
    _sRunMode: Literal['train', 'infer']  = 'infer'
    _sFrameMode: Literal['image', 'frame'] = 'frame'
    _iBatchSize: int = 16
    _iWidth: int = 640
    _iHeight: int = 480
    _tDevice: torch.device = torch.device('cpu')


    _TrainDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sTrainPath),
        batch_size=_iBatchSize,
        shuffle=True
    )
    _ValDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sValPath),
        batch_size=_iBatchSize,
        shuffle=True
    )
    _TestDataLoader: DataLoader = DataLoader(
        DataClass(sJsonBasePath=_sTestPath),
        batch_size=_iBatchSize,
        shuffle=True
    )
    _BufferStream = BufferedInputStream(
        sPath=_sVideoPath,
        iWidth=_iWidth,
        iHeight=_iHeight,
        device=_tDevice
    )

    # _Model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    _Model = None

    runner = Runner(
        iBatchSize=_iBatchSize,
        TrainDataLoader=_TrainDataLoader,
        ValDataLoader=_ValDataLoader,
        TestDataLoader=_TestDataLoader,
        Model=_Model,
        Loss=torch.nn.CrossEntropyLoss(),
        Optimizer=torch.optim.Adam(_Model.parameters(), lr=1e-3) if _Model is not None else None
    )

    runner.run(BufferStream=_BufferStream, sRunMode=_sRunMode, iEpochLen=_iEpochLen)



