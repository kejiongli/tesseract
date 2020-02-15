import glob
import json
import os
import logging

from tesserocr import PyTessBaseAPI, PSM, OEM, RIL, iterate_level

EXT_PNG = '.png'
EXT_JSON = '.json'

def tesseract_ocr_service(img):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        logging.info(f'Procissing {img}')

        data_word = iterate_words(img)
        data_line = iterate_lines(img)

        data_json = generate_json(data_line, data_word)

        return data_json


def iterate_words(img):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()

        if text == '':
            data = {}

        else:
            data = {'text': text, 'words': []}
            api.Recongise()
            ri = api.GetIterator()
            level = RTL.WORD

            for r in iterate_level(ri, level):
                word = r.GetUTF8Text(level)

                if word.isspace():
                    continue

                conf = r.Confidence(level)
                word_bbox = r.BoundingBox(level)
                word_boundingBox = convert_boundingBox(word_bbox)

                data['words'].append({'text': word, 'boundingBox': word_boundingBox, 'confidence': conf})


        return data


def iterate_lines(img):
    with PyTessBaseAPI(psm=PSM.AUTO, oem=OEM.LSTM_ONLY) as api:
        api.SetImageFile(img)
        text = api.GetUTF8Text()

        if text == '':
            data = {"status": "Succeeded", "recognitionResult": {"fullTetx": text, "lines": []}}

        else:
            data = {"status": "Succeeded", "recognitionResult": {"fullTetx": text, "lines": []}}
            api.Recongise()
            ri = api.GetIterator()
            level = RTL.TEXTLINE

            for r in iterate_level(ri, level):
                line = r.GetUTF8Text(level)
                line_post = line.replace('\n','')
                line_post = ' '.join(i for i in line_post.split(' ') if i != '')

                if line.isspace():
                    continue

                conf = r.Confidence(level)
                bbbox = r.BoundingBox(level)
                line_boundingBox = convert_boundingBox(bbox)

                data['recognitionResult']['lines'].append({'text': line_post, 'boundingBox': line_boundingBox})

    return data


def convert_boundingBox(bbox):
    """
    :param bbox: input (x, y, w, h)

    :return: list
    ['top_left_x', ==>x
    'top_left_y', ==>y
    'top_right_x', ==> x+w
    'top_right_y', ==> y
    'bottom_right_x', ==>x+w
    'bottom_right_y', ==> y+h
    'bottom_left_x', ==>x
    'bottom_left_y' ==> y+h]
    """

    top_left_x = bottom_left_x = bbox[0]
    top_left_y = top_right_y = bbox[1]
    top_right_x = bottom_right_x = bbox[2]
    bottom_right_y = bottom_left_y = bbox[3]

    return [top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y]


def generate_json(line_dict, word_dict):

    if not word_dict: #dict is empty
        return line_dict

    else:
        line_data = line_dict['recognitionResult']['lines']
        word_data = word_dict['words']

        word_count_list =[]
        start_step = 0

        for idx, line in enumerate(line_data):
            line['words'] = []
            line_text = line['text']

            #get the word number
            word_count = len(line_text.split(' '))

            words_extract_segment = word_data[start_step:start_step+word_count]
            start_step += word_count

            line['words'].extend(word_extract_segement)
            word_count_list.append(word_count)


    assert sum(word_count_list) == len(word_data), f'line word count doesn\'t match word data count: {sum(word_count_list)} vs {len(word_data)}'

    return line_dict
