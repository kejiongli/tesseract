import os
import json

OUTPUT_JSON = 'tesseract-ocr.json'


def aggregate_json(in_files, file_name):
    out_file = os.path.join(file_name, OUTPUT_JSON)

    word_count = 0
    word_confidence = 0
    pages =[]

    for idx, page_file in enumerate(in_files):
        with open(page_file, 'r') as f:
            data = json.load(f)["recognitionResult"]

        page_data = {
            "pageNo": idx,
            "fullText": data['fullText'],
            "lines": data['lines']
        }

        pages.append(page_data)

        for line in data['lines']:
            for word in line['words']:
                word_count += 1
                word_confidence += word['confidence']


    letter_data = {
        "pages" : pages,
        "confidence": word_confidence/word_count,
        "metadata": {"total_pages": len(in_files), "ocr_error": 0}
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(letter_data, f, indent=4, ensure_ascii=False)