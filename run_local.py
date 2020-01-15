import glob
import json
import os

from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level


def func1(img: str):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        api.SetImageFile(img)

        text = api.GetUTF8Text()
        data = {'text': text, 'lines': []}

        lines = api.GetTextlines(raw_image=True, raw_padding=5)
        # lines = api.GetComponentImages(level=RIL.TEXTLINE, text_only=False, raw_image=True, raw_padding=5)
        for idx, [image, bbox, bid, pid] in enumerate(lines):
            # with open(os.path.join(odir, f'line-{idx}.jpg'), 'wb') as f:
            #     image.save(f)
            api.SetImage(image)
            text = api.GetUTF8Text()

            line_data = dict(bbox=bbox, bid=bid, pid=pid, text=text)

            # words = api.GetWords()
            # words = api.GetComponentImages(level=RIL.WORD, text_only=False, raw_image=True, raw_padding=5)
            # line_data['words'] = []
            # for iw, [image, bbox, _, _] in enumerate(words):
            #     with open(os.path.join(odir, f'line-{idx}-word{iw}.jpg'), 'wb') as f:
            #         image.save(f)
            #     api.SetImage(image)
            #     word = api.GetUTF8Text()
            #     line_data['words'].append({
            #         'text': word,
            #         'bbox': bbox
            #     })

            data['lines'].append(line_data)

        return data


def iterate_words(img):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        data = {'text': text, 'words': []}
        api.Recognize()
        ri = api.GetIterator()
        level = RIL.WORD
        for r in iterate_level(ri, level):
            word = r.GetUTF8Text(level)
            conf = r.Confidence(level)
            bbox = r.BoundingBox(level)
            print(conf, word, bbox)
            data['words'].append({'text': word, 'bbox': bbox})
        return data


def iterate_lines(img):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()
        data = {'text': text, 'lines': []}
        api.Recognize()
        ri = api.GetIterator()
        level = RIL.TEXTLINE
        for r in iterate_level(ri, level):
            line = r.GetUTF8Text(level)
            conf = r.Confidence(level)
            bbox = r.BoundingBox(level)
            print(conf, line, bbox)
            data['lines'].append({'text': line, 'bbox': bbox})
        return data


if __name__ == '__main__':
    idir = 'raw-imgs'
    odir = 'ocr-results'

    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        for img in glob.glob(os.path.join(idir, '*.png')):

            # data = iterate_words(img)

            data = iterate_lines(img)

            with open(os.path.join(odir, 'all_lines.json'), 'w') as f:
                json.dump(data, f, indent=4)

            break

