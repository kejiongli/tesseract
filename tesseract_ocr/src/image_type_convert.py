import os

from PIL import Image, ImageSequence

EXT_PNG = '.png'

def tiff_to_png(in_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    im = Image.open(in_file)
    for i, page in enumerate(ImageSequence.Iterator(im)):
        png_file_name = "Page-{:0>2d}.{}".format(i, EXT_PNG)
        png_path = os.path.join(out_dir, png_file_name)
        page.save(png_path)
