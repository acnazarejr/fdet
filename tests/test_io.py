"""Tests over io functions"""

#pylint: skip-file

import tempfile
import os
import pytest
import cv2
import json
import numpy as np
from fdet.utils.errors import DetectorIOError
import fdet

@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')

@pytest.fixture
def low_image(resources_path):
    low_path = os.path.join(resources_path, 'images', 'low.jpg')
    return cv2.cvtColor(cv2.imread(low_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)



def test_read_as_rgb(resources_path, low_image):
    with pytest.raises(DetectorIOError):
        fdet.io.read_as_rgb('invalid_path')
    image = fdet.io.read_as_rgb(os.path.join(resources_path, 'images', 'low.jpg'))
    assert image.ndim == 3
    assert image.shape == (400, 600, 3)
    assert np.array_equal(low_image, image)
    image = fdet.io.read_as_rgb(os.path.join(resources_path, 'images', 'gray.jpg'))
    assert image.ndim == 3
    assert image.shape == (400, 600, 3)

def test_save(low_image):
    with pytest.raises(DetectorIOError):
        fdet.io.save('invalid_dir/image', low_image)
    with tempfile.TemporaryDirectory() as temp_dir:
        path_to_save = os.path.join(temp_dir, 'out.jpg')
        with pytest.raises(DetectorIOError):
            fdet.io.save(path_to_save, np.asarray([10]))
        fdet.io.save(path_to_save, low_image)


def test_video_handle(resources_path):
    with pytest.raises(DetectorIOError):
        fdet.io.VideoHandle('invalid_path')
    with pytest.raises(DetectorIOError):
        fdet.io.VideoHandle(os.path.join(resources_path, 'images', 'low.jpg'))
    video = fdet.io.VideoHandle(os.path.join(resources_path, 'video.mp4'))
    assert len(video) == 49
    count = 0
    for idx, frame in video:
        count += 1
        assert count == idx
        assert frame.shape == (720, 1280, 3)


def test_draw(resources_path):
    detections = json.load(open(os.path.join(resources_path, 'outputs', 'mtcnn', 'low.json'), 'r'))
    low_image = os.path.join(resources_path, 'images', 'low.jpg')
    fdet.io.draw_detections(low_image, detections['low.jpg'], thickness=None)
