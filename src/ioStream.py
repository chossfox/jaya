import cv2
import torch
import torchvision.transforms as T


class BufferedInputStream:
    def __init__(
            self,
            sPath = 0,
            iWidth = 640,
            iHeight = 480,
            device=torch.device('cpu')
    ):
        self.Stream = cv2.VideoCapture(sPath)
        self.iWidth = iWidth
        self.iHeight = iHeight
        self.Device = device
        self.is_bRealTime = False if type(sPath) == str else True
        if self.is_bRealTime:
            self.Stream.set(cv2.CAP_PROP_FRAME_WIDTH, iWidth)
            self.Stream.set(cv2.CAP_PROP_FRAME_HEIGHT, iHeight)

    def get_frame(self, frame_num = 16):
        tBuffer = torch.tensor([], device=self.Device)
        transformList = [T.ToTensor()]
        if not self.is_bRealTime:
            transformList.append(T.Resize((self.iHeight, self.iWidth)))
        tCompose = T.Compose(transformList)
        for _ in range(frame_num):
            if self.Stream.isOpened():
                bRet, nFrame = self.Stream.read()
                if bRet:
                    tFrame = torch.unsqueeze(tCompose(nFrame), dim=0).to(self.Device)
                    tBuffer = torch.cat((tBuffer, tFrame), dim=0)

        return tBuffer

    def is_video_open(self):
        return self.Stream.isOpened()
