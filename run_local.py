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
        data = {'status': 'Succeed', 'recognitionResult': {'lines': []}}
        api.Recognize()
        ri = api.GetIterator()
        level = RIL.TEXTLINE
        for r in iterate_level(ri, level):
            line = r.GetUTF8Text(level)

            if line.isspace():
                continue

            conf = r.Confidence(level)
            bbox = r.BoundingBox(level)
            boundingBox = convert_boundingBox(bbox)

            generate_word_segement(line, bbox)

            data['recognitionResult']['lines'].append({'text': line, 'boundingBox': boundingBox})

        return data


def convert_boundingBox(bbox):
    top_left_x = bottom_left_x = bbox[0]
    top_left_y = top_right_y = bbox[1]
    bottom_right_x = top_right_x = bbox[2]
    bottom_left_y = bottom_right_y = bbox[3]

    return [top_left_x,top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y]


# def generate_word_segement(line, bbox):
#     char_len = len(list(line))
#     line_box_len = bbox[3]-bbox[0] #w-x
#     char_ space = int(line_box_len/char_len)


if __name__ == '__main__':
    idir = 'raw-imgs'
    odir = 'ocr-results'

    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        for img in glob.glob(os.path.join(idir, '*.png')):

            # data = iterate_words(img)

            data = iterate_lines(img)

            with open(os.path.join(odir, 'all_lines.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)

            break

