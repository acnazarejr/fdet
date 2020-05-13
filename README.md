# FDet - Deep Learning Face Detection

![Build](https://github.com/acnazarejr/fdet/workflows/Build/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/acnazarejr/fdet/badge)](https://www.codefactor.io/repository/github/acnazarejr/fdet)
[![codecov](https://codecov.io/gh/acnazarejr/fdet/branch/master/graph/badge.svg)](https://codecov.io/gh/acnazarejr/fdet)
![PyPI](https://img.shields.io/pypi/v/fdet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fdet)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/acnazarejr/fdet)
[![GitHub](https://img.shields.io/github/license/acnazarejr/fdet)](https://github.com/acnazarejr/fdet/blob/master/LICENSE)

The `fdet` is a ready-to-use implementation of deep learning face detectors for PyTorch in Python 3.5 or higher.

![Example](https://github.com/acnazarejr/fdet/raw/readme/example.jpg)

Despite the availability of implementations of these algorithms, there are some disadvantages we found when using them. So we create this project to offer these features:

- Real-time face detection;
- Support to batch detection (useful for fast detection of multiple images and videos);
- Ease of use through python library or command-line app;
- Provide a unified interface to assign 'CPU' or 'GPU' devices;
- Multiple GPU's support;
- On-demand and automatic model weights download.

Currently, there are two different detectors available on FDet:

- **MTCNN** - Joint face detection and alignment using multitask cascaded convolutional networks [[zhang:2016]](#references)
- **RetinaFace** - Single-stage dense face localisation in the wild. [[deng:2019]](#references)

It was written based on the other available implementations ([see credits](#credits)).

## Installation

1. **This implementation requires PyTorch**. If this is the first time you use PyTorch, please install it in your environment following the [oficial instructions](https://pytorch.org/get-started/locally/).

2. After install pytorch, `fdet` can be installed through pip:

```bash
pip install fdet
```

> The current version works on Windows, Linux and MacOs systems with python 3.5 or higher.

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

## Credits

This implementation is heavily inspired by:

- [TropComplique/mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch/)
- [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [ipazc/mtcnn](https://github.com/ipazc/mtcnn)

## References

- **[zhang:2016]**: Zhang, K., Zhang, Z., Li, Z. and Qiao, Y. (2016). *Joint face detection and alignment using multitask cascaded convolutional networks*. IEEE Signal Processing Letters, 23(10), 1499-1503. [(link to paper)](https://ieeexplore.ieee.org/abstract/document/7553523)

- **[deng:2019]**: Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I. and Zafeiriou, S. (2019). *Retinaface: Single-stage dense face localisation in the wild*. arXiv preprint arXiv:1905.00641. [(link to paper)](https://arxiv.org/abs/1905.00641)

## License

[MIT license](https://github.com/acnazarejr/fdet/blob/master/LICENSE)
