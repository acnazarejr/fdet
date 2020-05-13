# FDet - Deep Learning Face Detector

![Build](https://github.com/acnazarejr/fdet/workflows/Build/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/acnazarejr/fdet/badge)](https://www.codefactor.io/repository/github/acnazarejr/fdet)
[![codecov](https://codecov.io/gh/acnazarejr/fdet/branch/master/graph/badge.svg)](https://codecov.io/gh/acnazarejr/fdet)
![PyPI](https://img.shields.io/pypi/v/fdet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fdet)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/acnazarejr/fdet)
[![GitHub](https://img.shields.io/github/license/acnazarejr/fdet)](https://github.com/acnazarejr/fdet/blob/master/LICENSE)

The `fdet` is an easy to use face detection module based on MTCNN and RetinaFace algorithms.

## Installation

1. **This implementation requires PyTorch**. If this is the first time you use PyTorch, please install it in your environment following the [oficial instructions](https://pytorch.org/get-started/locally/).

2. After install pytorch, `fdet` can be installed through pip:

```bash
pip install fdet
```

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

## License

[MIT license](https://github.com/acnazarejr/fdet/blob/master/LICENSE)
