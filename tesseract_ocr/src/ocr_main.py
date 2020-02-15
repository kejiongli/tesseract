import glob
import json
import os
import logging


from tesseract_ocr_service import tesseract_ocr_service
from ocr_image_preprocessor import sharpen_image
from image_type_convert import tiff_to_png
from generate_ocr_output import aggregate_json

EXT_TIF = '.tif'
EXT_JSON = '.json'
EXT_PNG = '.png'

DIR_PNG = 'step1_png'
DIR_SHARPEN = 'step2_sharpen'
DIR_OCR = 'step3_ocr_json'


def run_ocr_process(top_dir, out_dir):
    for root, dirs, files in os.walk(top_dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            file_name = os.path.splitext(file_path)[0]
            file_ext = os.path.splitext(file_path)[1].lower()

            fname = name.split('.')[0]
            fname_dir = os.path.join(out_dir, fname)
            os.makedirs(fname_dir, exist_ok=True)

            logging.info(f"Step 1 - Generating png for {file_name}")
            png_dir = os.path.join(out_dir, fname, DIR_PNG)
            tiff_to_png(file_path, png_dir)

            logging.info(f"Step 2 - Sharpening png for {file_name}")
            png_sharpened_dir = os.path.join(fname_dir, DIR_SHARPEN)
            os.makedirs(png_sharpened_dir, exist_ok=True)

            for png_file in glob.glob(os.path.join(png_dir, f'*.{EXT_PNG}')):
                name = os.path.basename(png_file)
                sharpen_image(png_file, os.path.join(png_sharpened_dir, name))


            logging.info(f"Step 3 - Use OCR to get json from png {file_name}")
            ocr_json_dir = os.path.join(fname_dir, DIR_OCR)
            os.makedirs(ocr_json_dir, exist_ok=True)

            for png_file in glob.glob(os.path.join(png_sharpened_dir, f'*.{EXT_PNG}')):
                result = tesseract_ocr_service(png_file)
                name, _ = os.path.splitext(os.path.basename(png_file))

                name += EXT_JSON

                with open(os.path.join(ocr_json_dir, name), 'w') as f:
                    json.dump(result, f, intent=2, ensure_ascii=False)


            logging.info("Step 4 - Aggregate json {file_name}")
            json_files = sorted(glob.glob(os.path.join(ocr_json_dir, '*.{EXT_JSON}')))
            aggregate_json(json_files, file_name)

if __name__ == '__main__':
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DATA_DIR = os.path.join(CUR_DIR, '..', 'raw-imgs')
    OCR_OUTPUT_DIR = os.path.join(CUR_DIR, '..', 'ocr-results')
    run_ocr_process(AUDIO_DATA_DIR, OCR_OUTPUT_DIR)
