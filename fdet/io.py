"""io module"""

from typing import Tuple, Union, List, Sequence, Dict, Any
import cv2
import numpy as np
from colour import Color

class VideoHandle():
    """Help class to iterate over video"""

    def __init__(self, source: str) -> None:

        self._video_reader = cv2.VideoCapture(source)
        self._n_frames = 0
        while self._video_reader.grab():
            self._n_frames += 1
        self._video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)


    def __iter__(self) -> 'VideoHandle':
        self._video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self) -> Tuple[int, np.ndarray]:
        ret, image = self._video_reader.read()
        if ret:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return int(self._video_reader.get(cv2.CAP_PROP_POS_FRAMES)), image
        self._video_reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        raise StopIteration

    def __len__(self) -> int:
        return self._n_frames

def read_as_rgb(path: str) -> np.ndarray:
    """Read an image as RGB format"""
    return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def save(path: str, image: np.ndarray) -> None:
    """Save an image"""
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


ConfType = float
ValueType = Union[float, int]
PointType = Tuple[ValueType, ValueType]
BoxType = Tuple[ValueType, ValueType, ValueType, ValueType]

def _draw_bbox(image: np.ndarray, bbox: BoxType, color: Union[Color, str, tuple] = Color('red'),
               thickness: int = None) -> np.ndarray:
    """draw_bbox"""
    image = np.ascontiguousarray(image)
    if thickness is None:
        thickness = max(int(min(image.shape[0], image.shape[1])/100), 1)
    color = Color(color).rgb
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    p_x, p_y, width, height = tuple(bbox)
    p_x, p_y, width, height = int(p_x), int(p_y), int(width), int(height)
    return cv2.rectangle(image, (p_x, p_y), (p_x+width, p_y+height), color, thickness)

def _draw_points(image: np.ndarray, points: Sequence[PointType],
                 color: Union[Color, str] = Color('red'), thickness: int = None) -> np.ndarray:
    """draw_bbox"""
    image = np.ascontiguousarray(image)
    if thickness is None:
        thickness = max(int(min(image.shape[0], image.shape[1])/100), 2)
    color = Color(color).rgb
    color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    for point in points:
        image = cv2.circle(image, (int(point[0]), int(point[1])), thickness, color, -1)
    return image

def _draw_detection(image: np.ndarray, detection: Dict[str, Any],
                    color: Union[Color, str, tuple] = Color('red'),
                    thickness: int = None) -> np.ndarray:
    """draw_detection"""
    image = _draw_bbox(image, detection['box'], color=color, thickness=thickness)
    image = _draw_points(image, detection['keypoints'].values(), color=color, thickness=thickness)
    return image

def draw_detections(image: Union[np.ndarray, str], detections: List[Dict[str, Any]],
                    color: Union[Color, str, tuple] = Color('red'),
                    thickness: int = 2) -> np.ndarray:
    """draw_detections"""
    if isinstance(image, str):
        image = read_as_rgb(image)

    for detection in detections:
        image = _draw_detection(image, detection, color=color, thickness=thickness)

    return image
