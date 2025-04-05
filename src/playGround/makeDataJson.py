import os
import os.path as osp
import json
from dataclasses import dataclass, asdict
from PIL import Image
import random
from tqdm import tqdm
from typing import Dict, List
from data import Label, ClassColor
import numpy as np
import cv2
from copy import deepcopy


def load_json(path: str, encoding: str = 'cp949'):
    with open (path, 'r', encoding=encoding) as f:
        return json.load(f)


def makeSampleData(
        sFilePath: str,
        iTotalClassNum: int
)-> Dict[str, List[Label]]:
    iWidth, iHeight = Image.open(sFilePath).size
    iClassLen = random.randint(1, iTotalClassNum)
    lClassSelect = random.choices(range(iTotalClassNum), k=iClassLen)
    lDataList = []
    for iClassNum in lClassSelect:
        x: int = random.randint(0, iWidth - 128)
        y: int = random.randint(0, iHeight - 128)
        w: int = random.randint(128, min(256, iWidth - x))
        h: int = random.randint(128, min(256, iHeight - y))
        cData = Label(
            sFileName=osp.basename(sFilePath),
            iX=x,
            iY=y,
            iW=w,
            iH=h,
            iLabel=iClassNum,
            tColor=getattr(ClassColor, f't{iClassNum}'),
            sEncodingTest='012345가나다abcdef'
        )
        lDataList.append(asdict(cData))

    dDataDict = {}
    dDataDict['sFilePath'] = sFilePath
    dDataDict['lDataList'] = lDataList

    return dDataDict


def makeBoxView(sOutPath: str, dDataDict: Dict[str, List[Label]]):
    sOutPath: str = osp.join(osp.dirname(sOutPath), 'boxView')
    nInImage: np.ndarray = np.array(Image.open(dDataDict['sFilePath']).convert('RGB'))
    nOutImage: np.ndarray = deepcopy(nInImage)
    for dData in dDataDict['lDataList']:
        dData: dict
        nOutImage = cv2.rectangle(
            nInImage,
            (dData['iX'], dData['iY']),
            (dData['iX'] + dData['iW'], dData['iY'] + dData['iH']),
            dData['tColor'],
            2
        )

    os.makedirs(sOutPath, exist_ok=True)
    Image.fromarray(nOutImage).save(osp.join(sOutPath, osp.basename(dDataDict['sFilePath'])))


if __name__ == '__main__':
    _sInPath = rf'G:\data\test\images'
    _sOutPath = rf'G:\data\test\label'
    _lFileList = os.listdir(_sInPath)
    _iTotalClassNum = 5

    _pBar = tqdm(_lFileList, total=len(_lFileList))
    for _sFile in _lFileList:
        _sExt = osp.splitext(_sFile)[-1].lower()
        if not _sExt in Image.registered_extensions():
            continue

        _pBar.set_description_str(f'[makeDataJson][{_sFile}]')
        _dDataDict = makeSampleData(sFilePath=osp.join(_sInPath, _sFile), iTotalClassNum=_iTotalClassNum)

        os.makedirs(_sOutPath, exist_ok=True)
        with open(osp.join(_sOutPath, f'{osp.splitext(_sFile)[0]}.json'), 'w') as f:
            json.dump(_dDataDict, f, ensure_ascii=False, indent=4)

        makeBoxView(sOutPath=_sOutPath, dDataDict=_dDataDict)

        _pBar.update()


