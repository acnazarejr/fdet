"""Tests over MTCNN detector"""

#pylint: skip-file

import pytest
from fdet.utils.errors import DetectorValueError
from fdet import MTCNN

def test_mtcnn_raises_invalid_min_size():
    with pytest.raises(DetectorValueError):
        MTCNN(min_face_size=-1)
    with pytest.raises(DetectorValueError):
        MTCNN(min_face_size=10)
    with pytest.raises(DetectorValueError):
        MTCNN(min_face_size=1050)
    MTCNN(min_face_size=50)

def test_mtcnn_raises_invalid_thresholds():
    with pytest.raises(DetectorValueError):
        MTCNN(thresholds=0.5)
    with pytest.raises(DetectorValueError):
        MTCNN(thresholds=(0.5, 0.5))
    with pytest.raises(DetectorValueError):
        MTCNN(thresholds=(0.5, 0.5, 2.0))
    with pytest.raises(DetectorValueError):
        MTCNN(thresholds=(-1.0, 0.5, 0.5))
    MTCNN(thresholds=(0.5, 0.5, 0.7))

def test_mtcnn_raises_invalid_nms_thresholds():
    with pytest.raises(DetectorValueError):
        MTCNN(nms_thresholds=0.5)
    with pytest.raises(DetectorValueError):
        MTCNN(nms_thresholds=(0.5, 0.5))
    with pytest.raises(DetectorValueError):
        MTCNN(nms_thresholds=(0.5, 0.5, 2.0))
    with pytest.raises(DetectorValueError):
        MTCNN(nms_thresholds=(-1.0, 0.5, 0.5))
    MTCNN(nms_thresholds=(0.5, 0.5, 0.7))
