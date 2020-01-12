import glob
import os

from tesserocr import PyTessBaseAPI, PSM, OEM

if __name__ == '__main__':
    idir = 'raw-imgs'
    odir = 'ocr-results'

    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        for img in glob.glob(os.path.join(idir, '*.png')):
            api.SetImageFile(img)
            with open(os.path.join(odir, f'ocr-{os.path.basename(img)}.txt'), 'w', encoding='utf-8') as f:
                f.write(api.GetUTF8Text())
            print(api.AllWordConfidences())

