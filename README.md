1. Create virtual environment: `python -m venv venv`
1. Install `pillow` package 
   ```cmd
    pip install -r requirements.txt
    ```
1. Download `tesserocr` whl file from [here](https://github.com/simonflueckiger/tesserocr-windows_build/releases) and install using pip
    ```$xslt
    pip install <downloadfile>.whl
    ```
1. Download data file `https://github.com/tesseract-ocr/tessdata/blob/master/eng.traineddata` and save the file in `$PATH_TO_PYTHON/tessdata`, 
where `$PATH_TO_PYTHON` can be obtained by run cmd `where python`
