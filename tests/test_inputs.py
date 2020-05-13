"""Test detection methods"""

#pylint: skip-file

import json
import pytest
import os
import cv2
import numpy as np
import fdet
from fdet.utils.errors import DetectorInputError

@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')

@pytest.fixture
def images(resources_path):
    images_dir = os.path.join(resources_path, 'images')
    images_buffer = dict()
    images_buffer['low'] = fdet.io.read_as_rgb(os.path.join(images_dir, 'low.jpg'))
    images_buffer['medium'] = fdet.io.read_as_rgb(os.path.join(images_dir, 'medium.jpg'))
    return images_buffer

@pytest.fixture
def mtcnn_detector():
    return fdet.MTCNN(cuda_enable=False)

def test_detect_image_input(mtcnn_detector):
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(None)
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(1111)
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect('invalid_input')
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100, 4, 4)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100, 1)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100)))

def test_detect_batch_input(mtcnn_detector, images):
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(None)
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(111)
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect([])
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect('invalid_input')
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(np.zeros((100, 100, 3)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect([np.zeros((100, 100, 1)), np.zeros((100, 100, 1))])
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(images['low'])
    with pytest.raises(Exception):
        mtcnn_detector.batch_detect([images['low'], images['medium']])

