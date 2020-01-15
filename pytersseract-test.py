from PIL import Image
from tesserocr import PyTessBaseAPI, PSM, OEM, RIL
import glob
import os

odir = 'ocr-results'
idir = 'raw-images'

if __name__ == '__main__':
    with PyTessBaseAPI() as api:
        for img in glob.glob(os.path.join(idir, '*.png')):
            api.SetImage(img)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            print('Found {} textline image components.'.format(len(boxes)))
            for i, (im, box, _, _) in enumerate(boxes):
                # im is a PIL image object
                # box is a dict with x, y, w and h keys
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocrResult = api.GetUTF8Text()
                conf = api.MeanTextConf()
                print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                      "confidence: {1}, text: {2}".format(i, conf, ocrResult, **box))
