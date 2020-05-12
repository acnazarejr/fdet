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
    images_buffer['high'] = fdet.io.read_as_rgb(os.path.join(images_dir, 'high.jpg'))
    images_buffer['gray'] = fdet.io.read_as_rgb(os.path.join(images_dir, 'gray.jpg'))
    return images_buffer

@pytest.fixture
def batch(resources_path):
    batch_images = list()
    for frame_file in os.listdir(os.path.join(resources_path, 'frames')):
        batch_images.append(fdet.io.read_as_rgb(os.path.join(resources_path, 'frames', frame_file)))
    return batch_images


@pytest.fixture
def mtcnn_outputs(resources_path):
    outputs_dir = os.path.join(resources_path, 'outputs', 'mtcnn')
    outputs = dict()
    outputs['low'] = json.load(open(os.path.join(outputs_dir, 'low.json'), 'r'))['low.jpg']
    outputs['medium'] = json.load(open(os.path.join(outputs_dir, 'medium.json'), 'r'))['medium.jpg']
    outputs['high'] = json.load(open(os.path.join(outputs_dir, 'high.json'), 'r'))['high.jpg']
    outputs['gray'] = json.load(open(os.path.join(outputs_dir, 'gray.json'), 'r'))['gray.jpg']
    outputs['frames'] = json.load(open(os.path.join(outputs_dir, 'frames.json'), 'r'))
    outputs['frames'] = list(outputs['frames'].values())
    return outputs

@pytest.fixture
def mtcnn_detector():
    return fdet.MTCNN(cuda_enable=False)

def test_detect_image_input(mtcnn_detector, images, mtcnn_outputs):
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect('invalid_input')
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100, 4, 4)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100, 1)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.detect(np.zeros((100, 100)))
    low_detections = mtcnn_detector.detect(images['low'])
    assert low_detections == mtcnn_outputs['low']

def test_detect_batch_input(mtcnn_detector, batch, images, mtcnn_outputs):
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect('invalid_input')
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(np.zeros((100, 100, 3)))
    with pytest.raises(DetectorInputError):
        mtcnn_detector.batch_detect(images['low'])
    batch_detections = mtcnn_detector.batch_detect(batch)
    assert batch_detections == mtcnn_outputs['frames']

