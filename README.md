# FDet - Deep Learning Face Detector

The `fdet` is an easy to use face detection module based on MTCNN and RetinaFace algorithms.




## Installation

Currently it is supported Python 3.5 or higher. It can be installed through pip:

```
pip install fdet
```
> **IMPORTANT**: This implementation requires PyTorch>=1.10. If this is the first time you use PyTorch, please install it in your environment following the [oficial instructions](https://pytorch.org/get-started/locally/).


## Quick Start

```python
>> from fdet import io, RetinaFace

>> image = io.read_as_rgb('path_to_image.jpg')
>> detector = RetinaFace(backbone='MOBILENET')
>> detector.detect(image)
[{'box': [511, 47, 35, 45],
  'confidence': 0.9999996423721313,
  'keypoints': {'left_eye': [517, 70],
                'mouth_left': [522, 87],
                'mouth_right': [531, 83],
                'nose': [520, 77],
                'right_eye': [530, 65]}}]
```
