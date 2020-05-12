"""Tests over MTCNN detector"""

#pylint: skip-file

import os
import json
import pytest
from fdet.utils.errors import DetectorValueError
import fdet

@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')

@pytest.fixture
def outputs_dir(resources_path):
    return os.path.join(resources_path, 'outputs')

@pytest.fixture
def images_dir(resources_path):
    return os.path.join(resources_path, 'images')

@pytest.fixture
def resnet_detector():
    return fdet.RetinaFace(backbone='RESNET50', cuda_enable=False)

@pytest.fixture
def mobilenet_detector():
    return fdet.RetinaFace(backbone='MOBILENET', cuda_enable=False)

def test_retinaface_raises_invalid_backbone():
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(backbone=20)
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(backbone='xxxxx')
    fdet.RetinaFace(backbone='RESNET50')
    fdet.RetinaFace(backbone='MOBILENET')

def test_retinaface_raises_invalid_max_size():
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(max_face_size=-1)
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(max_face_size=10)
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(max_face_size=1050)
    fdet.RetinaFace(backbone='MOBILENET', max_face_size=200)

def test_retinaface_raises_invalid_threshold():
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(threshold=2.0)
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(threshold=-1.0)
    fdet.RetinaFace(backbone='MOBILENET', threshold=0.8)

def test_retinaface_raises_invalid_nms_threshold():
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(nms_threshold=2.0)
    with pytest.raises(DetectorValueError):
        fdet.RetinaFace(nms_threshold=-1.0)
    fdet.RetinaFace(backbone='MOBILENET', nms_threshold=0.4)


def test_retinaface_low(resnet_detector, mobilenet_detector, images_dir, outputs_dir):
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'low.jpg'))

    gt_path = os.path.join(outputs_dir, 'retinaface_resnet50', 'low.json')
    assert json.load(open(gt_path, 'r'))['low.jpg'] == resnet_detector.detect(image)

    gt_path = os.path.join(outputs_dir, 'retinaface_mobilenet', 'low.json')
    assert json.load(open(gt_path, 'r'))['low.jpg'] == mobilenet_detector.detect(image)

def test_retinaface_medium(resnet_detector, mobilenet_detector, images_dir, outputs_dir):
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'medium.jpg'))

    gt_path = os.path.join(outputs_dir, 'retinaface_resnet50', 'medium.json')
    assert json.load(open(gt_path, 'r'))['medium.jpg'] == resnet_detector.detect(image)

    gt_path = os.path.join(outputs_dir, 'retinaface_mobilenet', 'medium.json')
    assert json.load(open(gt_path, 'r'))['medium.jpg'] == mobilenet_detector.detect(image)

def test_retinaface_gray(resnet_detector, mobilenet_detector, images_dir, outputs_dir):
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'gray.jpg'))

    gt_path = os.path.join(outputs_dir, 'retinaface_resnet50', 'gray.json')
    assert json.load(open(gt_path, 'r'))['gray.jpg'] == resnet_detector.detect(image)

    gt_path = os.path.join(outputs_dir, 'retinaface_mobilenet', 'gray.json')
    assert json.load(open(gt_path, 'r'))['gray.jpg'] == mobilenet_detector.detect(image)

def test_retinaface_sea(resnet_detector, mobilenet_detector, images_dir, outputs_dir):
    image = fdet.io.read_as_rgb(os.path.join(images_dir, 'sea.jpg'))

    gt_path = os.path.join(outputs_dir, 'retinaface_resnet50', 'sea.json')
    assert json.load(open(gt_path, 'r'))['sea.jpg'] == resnet_detector.detect(image)

    gt_path = os.path.join(outputs_dir, 'retinaface_mobilenet', 'sea.json')
    assert json.load(open(gt_path, 'r'))['sea.jpg'] == mobilenet_detector.detect(image)

def test_retinaface_batch(mobilenet_detector, resources_path, outputs_dir):

    batch_images = list()
    for frame_file in os.listdir(os.path.join(resources_path, 'frames')):
        batch_images.append(fdet.io.read_as_rgb(os.path.join(resources_path, 'frames', frame_file)))
    gt_path = os.path.join(outputs_dir, 'retinaface_mobilenet', 'frames.json')
    gt_detections = json.load(open(gt_path, 'r'))
    assert list(gt_detections.values()) == mobilenet_detector.batch_detect(batch_images)
