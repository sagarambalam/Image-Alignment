#!/usr/bin/env python
# coding: utf-8


import os
import sys
import argparse

import numpy as np
import cv2
import math
from scipy import ndimage

from PIL import Image
import pytesseract

import nltk
from nltk.tokenize import TreebankWordTokenizer

from pathlib import Path
import glob


def read_image(file_path):
    '''
    Input: string
    Output: numpy array
    Description: The function takes input as the file path of the image and converts
    into numpy array
    '''
    return cv2.imread(file_path)

def write_image(input_path, output_path, output_nd_array):
    '''
    Input: string, string, numpy array
    Output: Numpy array written into the image at the output filepath
    '''
    file_name = input_path.split('/')[-1].split('.')[0]
    extension = input_path.split('.')[-1]
    new_file_path = output_path+'/aligned_'+file_name+'.'+extension

    cv2.imwrite(new_file_path, output_nd_array)


def rgb_to_gray(rgb_nd_array):
    '''
    Input: nd array
    Output: nd array
    Description: The function takes into input RGB image object in the form of numpy array
    and converts it into a gray scale values of which are in between 0 and 1.
    '''
    return cv2.cvtColor(rgb_nd_array, cv2.COLOR_BGR2GRAY)


def get_edges(gray_scaled_image_array):
    '''
    Input: nd array
    Output: nd array
    Description: The function takes input gray scaled image array to be used for edge detection.
    Using Canny Edge detector with kernel size of 3, edges of each objects in the pixel are highlighted.
    '''
    sobel_kernel_size = 3
    min_threshold_value = 100
    max_threshold_value = 100

    return cv2.Canny(gray_scaled_image_array, min_threshold_value, max_threshold_value, apertureSize=sobel_kernel_size)


def hough_line_detection(image_edges):
    '''
    Input: nd array
    Output: list
    Description: The function takes input numpy array where edges are highlighted and finds lines using
    hough transform.
    '''
    rho = 1
    theta = math.pi / 180.0
    threshold = 100
    min_line_length = 100
    max_line_gap = 5

    return cv2.HoughLinesP(image_edges, rho, theta, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)


def get_angles(list_of_coordinates):
    '''
    Input: list
    Output: list
    Description: The finction takes input list of list with each list containing coordinates of lines and returns 
    list of angles of the lines
    '''

    return [math.degrees(math.atan2(y2 - y1, x2 - x1)) for [[x1, y1, x2, y2]] in list_of_coordinates]


def rotate_image(input_image, angle_of_rotation):
    '''
    Input: nd_array
    Output: float
    Description: The function returns rotated image based on the angle given for rotation.
    '''
    return ndimage.rotate(input_image, angle_of_rotation)


def detect_horizontal_orientation(image_nd_array):
    '''
    Input: nd_array
    Output: float
    Descriptiion: The function takes input as the numpy array of the input image and returns the final angle in which the
    image has to be rotated.
    '''

    final_angle = None

    rotation_angle_1 = 90.0
    rotation_angle_2 = -90.0

    rotated_image_angle_1 = rotate_image(image_nd_array, rotation_angle_1)
    rotated_image_angle_2 = rotate_image(image_nd_array, rotation_angle_2)

    ocr_text_rotated_image_angle_1 = pytesseract.image_to_string(
        Image.fromarray(np.uint8(rotated_image_angle_1)).convert('RGB'))
    ocr_text_rotated_image_angle_2 = pytesseract.image_to_string(
        Image.fromarray(np.uint8(rotated_image_angle_2)).convert('RGB'))

    nltk_words_corpus = set(nltk.corpus.words.words())
    tokenizer_obj = TreebankWordTokenizer()

    tokens_ocr_text_angle_1 = tokenizer_obj.tokenize(
        ocr_text_rotated_image_angle_1)
    tokens_ocr_text_angle_2 = tokenizer_obj.tokenize(
        ocr_text_rotated_image_angle_2)

    english_token_list_angle_1 = len(
        [word for word in tokens_ocr_text_angle_1 if word.lower() in nltk_words_corpus])
    english_token_list_angle_2 = len(
        [word for word in tokens_ocr_text_angle_2 if word.lower() in nltk_words_corpus])

    if english_token_list_angle_1 > english_token_list_angle_2:
        final_angle = rotation_angle_1
    else:
        final_angle = rotation_angle_2

    return final_angle


def get_response(message, is_success, correction_angle):
    if correction_angle:
        return {'message': message, 'success': is_success, 'correction_angle': round(correction_angle, 3)}
    else:
        return {'message': message, 'success': is_success, 'correction_angle': None}


def deskew(input_image):

    final_image = None
    try:
        gray_scaled_image = rgb_to_gray(input_image)
        image_edges = get_edges(gray_scaled_image)
        lines = hough_line_detection(image_edges)
        angles = get_angles(lines)

        if len(angles) < 5:
            final_image = input_image
            response = get_response(
                'Insufficient lines detected for orientation detection', False, None)

        else:
            median_angle = np.median(angles)
            print(median_angle)
            if abs(median_angle) != 90 and median_angle != 0:
                final_image = rotate_image(input_image, median_angle)
                response = get_response('Image misaligned', True, median_angle)

            elif median_angle == 0:
                final_image = input_image
                response = get_response('Image aligned', True, median_angle)

            elif abs(median_angle) == 90:
                corrected_angle = detect_horizontal_orientation(input_image)
                final_image = rotate_image(input_image, corrected_angle)
                response = get_response(
                    'Image misaligned', True, corrected_angle)

    except Exception as e:
        respose = get_response("Exception at deskew function", False, None)

    return response, final_image


def main(input_path, input_type, output_path):

    Path(output_path).mkdir(parents=True, exist_ok=True)
    if input_type == 'file':
        input_image = read_image(input_path)
        res, output_image = deskew(input_image)

        if res['success'] == True:
            write_image(input_path, output_path, output_image)
        print(res)
    else:
        for path in glob.glob(input_path+'/*'):
            print(path)
            input_image = read_image(path)
            res, output_image = deskew(input_image)

            if res['success'] == True:
                write_image(path, output_path, output_image)
            print(res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", dest="input_path",
                        help="Path to Input Image", required=True)
    parser.add_argument("--output", dest="output_dir",
                        help="Output Location", required=False,
                        default='Output')
    parser.add_argument("--input-type", dest="input_type",
                        help="Input type(file or dir)", required=False,
                        default='file', choices=['file', 'dir'])
    args = parser.parse_args()
    main(args.input_path, args.input_type, args.output_dir)
