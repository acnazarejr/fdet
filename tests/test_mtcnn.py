"""Tests over MTCNN detector"""

#pylint: skip-file

import os
import json
import pytest
import numpy as np
from fdet.utils.errors import DetectorValueError
import fdet

def _assert_detections(dets_a, dets_b):
    for det_a, det_b in zip(dets_a, dets_b):

        assert det_a['box'] == pytest.approx(det_b['box'], rel=0.5)

        key_a = np.asarray(list(det_a['keypoints'].values())).flatten()
        key_b = np.asarray(list(det_b['keypoints'].values())).flatten()
        assert key_a == pytest.approx(key_b, rel=0.5)

        assert det_a['confidence'] == pytest.approx(det_b['confidence'], rel=0.02)

@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')

@pytest.fixture
def outputs_dir(resources_path):
    return os.path.join(resources_path, 'outputs', 'mtcnn')

@pytest.fixture
def images_dir(resources_path):
    return os.path.join(resources_path, 'images')

@pytest.fixture
def mtcnn_detector():
    return fdet.MTCNN(cuda_enable=False)

def test_mtcnn_raises_invalid_min_size():
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(min_face_size=-1)
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(min_face_size=10)
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(min_face_size=1050)
    fdet.MTCNN(min_face_size=50)

def test_mtcnn_raises_invalid_thresholds():
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(thresholds=0.5)
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(thresholds=(0.5, 0.5))
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(thresholds=(0.5, 0.5, 2.0))
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(thresholds=(-1.0, 0.5, 0.5))
    fdet.MTCNN(thresholds=(0.5, 0.5, 0.7))

def test_mtcnn_raises_invalid_nms_thresholds():
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(nms_thresholds=0.5)
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(nms_thresholds=(0.5, 0.5))
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(nms_thresholds=(0.5, 0.5, 2.0))
    with pytest.raises(DetectorValueError):
        fdet.MTCNN(nms_thresholds=(-1.0, 0.5, 0.5))
    fdet.MTCNN(nms_thresholds=(0.5, 0.5, 0.7))

def test_mtcnn_low(mtcnn_detector, images_dir, outputs_dir):
    gt_detections = json.load(open(os.path.join(outputs_dir, 'low.json'), 'r'))['low.jpg']
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'low.jpg'))
    _assert_detections(gt_detections, mtcnn_detector.detect(image))

def test_mtcnn_medium(mtcnn_detector, images_dir, outputs_dir):
    gt_detections = json.load(open(os.path.join(outputs_dir, 'medium.json'), 'r'))['medium.jpg']
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'medium.jpg'))
    _assert_detections(gt_detections, mtcnn_detector.detect(image))

def test_mtcnn_gray(mtcnn_detector, images_dir, outputs_dir):
    gt_detections = json.load(open(os.path.join(outputs_dir, 'gray.json'), 'r'))['gray.jpg']
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'gray.jpg'))
    _assert_detections(gt_detections, mtcnn_detector.detect(image))

def test_mtcnn_sea(mtcnn_detector, images_dir, outputs_dir):
    gt_detections = json.load(open(os.path.join(outputs_dir, 'sea.json'), 'r'))['sea.jpg']
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'sea.jpg'))
    _assert_detections(gt_detections, mtcnn_detector.detect(image))

def test_mtcnn_batch(mtcnn_detector, resources_path, outputs_dir):

    batch_images = list()
    for frame_file in os.listdir(os.path.join(resources_path, 'frames')):
        batch_images.append(fdet.io.read_as_rgb(os.path.join(resources_path, 'frames', frame_file)))
    gt_detections = json.load(open(os.path.join(outputs_dir, 'frames.json'), 'r'))
    batch_detections = mtcnn_detector.batch_detect(batch_images)
    for gt_image_detections, image_detections in zip(gt_detections.values(), batch_detections):
        _assert_detections(gt_image_detections, image_detections)
