# Image-Alignment
## Problem Statement:
Given an scanned image, We want to identify if the image is correctly aligned with the horizontal and vertical axes.
If it is not aligned, we need to correct the alignment of the image.

## Approach:
1. We use Image processing techniques to identify the edges of the image and try to identify any lines on the image.
2. This has been achieved by using Hough Transform. Using hough transform on the image with edges highlighted, we get the lines
3. We try to identify the angle of these lines with respect to axes.
4. Based on this we are able to identify if the angle by which the image is misaligned.
5. The next part of challenge is to identify the orientation of the image.
6. To solve this, we use OCR to identify the possible orientation.
7. We rotate the image by 90 degrees on both the direction(clockwise and anticlockwise), we do an OCR and try to find out for which orientation more words were identified. This is considered as the final angle of rotation.

## Tools / Packages Used:
1. Python
2. OpenCV
3. Tesseract
4. NLTK
5. Pillow

## Features / Functionality:
1. The solution to the problem has been built in the form of a Command Line Interface.
2. The CLI application takes input as input file, output destination, and input type.
3. Input can be either an image file or a directory.
4. If the input is image file then image file will be processed and saved into the destination directory.
5. If the input is a directory, then all the images in the directory will be processed and saved into the destination directory.

## Installation instructions:
1. On Cloning the repository, the following command will help install all the necessary packages
  - pip install -r requirements.txt
2. We need to install tesseract into our system for OCR. The following steps have to be followed.
  - On Linux:
    - apt-get install tesseract-ocr
  - On Mac:
    - brew install tesseract

## Exceptions:
1. If input type is dir is provided, currently all the files will be processed. This can create issues for non image files.
2. Current algorithm fails to get orientation correct if the angle is more than 180% or less than 0% with respect to horizontal axis.

## Usage:
On Terminal:
python --input 'Your input path' --output 'Your output path' --input-type 'file or dir'
