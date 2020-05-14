# FDet - Deep Learning Face Detection

![Build](https://github.com/acnazarejr/fdet/workflows/Build/badge.svg)
[![CodeFactor](https://www.codefactor.io/repository/github/acnazarejr/fdet/badge)](https://www.codefactor.io/repository/github/acnazarejr/fdet)
[![codecov](https://codecov.io/gh/acnazarejr/fdet/branch/master/graph/badge.svg)](https://codecov.io/gh/acnazarejr/fdet)
![PyPI](https://img.shields.io/pypi/v/fdet)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fdet)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/acnazarejr/fdet)
[![GitHub](https://img.shields.io/github/license/acnazarejr/fdet)](https://github.com/acnazarejr/fdet/blob/master/LICENSE)

The `fdet` is a ready-to-use implementation of deep learning face detectors using PyTorch.

![Example](https://github.com/acnazarejr/fdet/raw/master/assets/example.jpg)

You can use it in two ways: Directly in your code, as a [python library](#python-library-usage) or by [command-line](#command-line-usage):

```python
>>> from fdet import io, RetinaFace
>>> detector = RetinaFace(backbone='RESNET50')
>>> detector.detect(io.read_as_rgb('path_to_image.jpg'))
[{'box': [511, 47, 35, 45], 'confidence': 0.9999996423721313,
  'keypoints': {'left_eye': [517, 70], 'right_eye': [530, 65], 'nose': [520, 77],
                'mouth_left': [522, 87], 'mouth_right': [531, 83]}}]
```

```bash
fdet retinaface -b RESNET50 -i path_to_image.jpg -o detections.json
```

## **Features**

Currently, there are two different detectors available on FDet:

- **MTCNN** - Joint face detection and alignment using multitask cascaded convolutional networks [[zhang:2016]](#references)
- **RetinaFace** - Single-stage dense face localisation in the wild. [[deng:2019]](#references). You can use it with two different backbones:
  - MobileNet: Fast and light-weighted model (achieves high FPS)
  - Resnet50: A medium-size model for better results, but slower.

Despite the availability of different implementations of these algorithms, there are some disadvantages we found when using them. So we create this project to offer the following features, in one package:

- :star: Real-time face detection;
- :star: Support for batch detection (useful for fast detection of multiple images and videos);
- :star: Ease of use through python library or command-line app;
- :star: Provide a unified interface to assign 'CPU' or 'GPU' devices;
- :star: Multiple GPU's support;
- :star: On-demand and automatic model weights download;
- :star: Compatible with Windows, Linux, and macOS systems.

## **Installation**

1. You need to [install PyTorch](https://pytorch.org/get-started/locally/) first (if you have a GPU, install PyTorch with CUDA support).

2. Then `fdet` can be installed through pip:

```bash
pip install fdet
```

## **Python Library Usage**

If you want to use `fdet` from python, just import it,

```python
import fdet
```

and instantiate your desired detector, with its respective parameters:

```python
fdet.MTCNN(min_face_size, thresholds, nms_thresholds, cuda_enable, cuda_devices, cuda_benchmark)
```

```python
fdet.MTCNN(min_face_size = 20.0, thresholds = (0.6, 0.7, 0.8), nms_thresholds=(0.7, 0.7, 0.7),
           cuda_enable=True, cuda_devices=None, cuda_benchmark=True)
```

`fdet.MTCNN(min_face_size, thresholds, nms_thresholds, cuda_enable, cuda_devices, cuda_benchmark)`

- `min_face_size` (float, optional): The minimum size of the face to detect. [default: 20.0].
- `thresholds` (tuple, optional): The thresholds fo each MTCNN step. [default: (0.6, 0.7, 0.8)]

    nms_thresholds (Tuple[float, float, float], optional): The NMS thresholds fo each MTCNN
        step. Defaults to (0.7, 0.7, 0.7).
    cuda_enable (bool, optional): Indicates if cuda should be used. Defaults to
        cuda.is_available().
    cuda_devices (Optional[List[int]], optional): CUDA GPUs to be used. If None, uses all
        avaliable GPUs. Defaults to None.
    cuda_benchmark (bool, optional): [description]. Indicates if the cuda_benchmark is
        enable or not. Defaults to True.





### **Singe-Image Detection**

The following example illustrates the ease of use of this library to detect faces in a single-image using MTCNN and to create an output image with all detected faces drawn.

```python
>>> from fdet import io, MTCNN
>>>
>>> detector = MTCNN()
>>> image = io.read_as_rgb('example.jpg')
>>>
>>> detections = detector.detect(image)
>>> print(detections)
[{'box': [511, 47, 35, 45], 'confidence': 0.9999996423721313,
  'keypoints': {'left_eye': [517, 70], 'right_eye': [530, 65], 'nose': [520, 77],
                'mouth_left': [522, 87], 'mouth_right': [531, 83]}}]
>>>
>>> output_image = io.draw_detections(image, detections, color='white', thickness=5)
>>> io.save('output.jpg', output_image)
```

The detector returns a list of `dict` objects. Each `dict` contains three main keys:

- `'box'`: The bounding box formatted as a list `[x, y, width, height]`;
- `'confidence'`: The probability for a bounding box to be matching a face;
- `'keypoints'`:
The five landmarks formatted into a `dict` with the keys `'left_eye'`, `'right_eye'`, `'nose'`, `'mouth_left'`, `'mouth_right'`. Each keypoint is identified by a pixel position `[x, y]`.

The `io.read_as_rgb()` is a wrapper for opencv `cv2.imread()` to ensure an RGB image and can be replaced by:

```python
image = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

### **Batch Detection**

The library is also capable of performing face detection on image batches, typically providing considerable speed-up. A batch should be structured as list of images (`numpy` arrays)  of equal dimension. The returned detections list will have an additional first dimension corresponding to the batch size. Each image in the batch may have one or more faces detected.

In the following example, we detect faces in every frame of a video using batchs of 10 images.

```python
>>> import cv2
>>> from fdet import io, RetinaFace
>>>
>>> BATCH_SIZE = 10
>>>
>>> detector = RetinaFace(backbone='MOBILENET', cuda_devices=[0,1])
>>> vid_cap = cv2.VideoCapture('path_to_video.mp4')
>>>
>>> video_face_detections = [] # list to store all video face detections
>>> image_buffer = [] # buffer to store the batch
>>>
>>> while True:
>>>
>>>     success, frame = vid_cap.read() # read the frame from video capture
>>>     if not success:
>>>         break # end of video
>>>
>>>     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert to RGB
>>>     image_buffer.append(frame) # add frame to buffer
>>>
>>>     if len(image_buffer) == BATCH_SIZE: # if buffer is full, detect the batch
>>>         batch_detections = detector.batch_detect(image_buffer)
>>>         video_face_detections.extend(batch_detections)
>>>         image_buffer.clear() # clear the buffer
>>>
>>> if image_buffer: # checks if images remain in the buffer and detect it
>>>     batch_detections = detector.batch_detect(image_buffer)
>>>     video_face_detections.extend(batch_detections)
```

## **Command-line Usage**

![Terminal](https://github.com/acnazarejr/fdet/raw/master/assets/terminal.gif)

## Credits

The FDet was written heavily inspired by the other available implementations ([see credits](#credits)).

- [TropComplique/mtcnn-pytorch](https://github.com/TropComplique/mtcnn-pytorch/)
- [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [ipazc/mtcnn](https://github.com/ipazc/mtcnn)

The current MTCNN version was implemented with the help of [Davi Beltrão](@Davibeltrao).

## References

- **[zhang:2016]**: Zhang, K., Zhang, Z., Li, Z. and Qiao, Y. (2016). *Joint face detection and alignment using multitask cascaded convolutional networks*. IEEE Signal Processing Letters, 23(10), 1499-1503. [(link to paper)](https://ieeexplore.ieee.org/abstract/document/7553523)

- **[deng:2019]**: Deng, J., Guo, J., Zhou, Y., Yu, J., Kotsia, I. and Zafeiriou, S. (2019). *Retinaface: Single-stage dense face localisation in the wild*. arXiv preprint arXiv:1905.00641. [(link to paper)](https://arxiv.org/abs/1905.00641)

## License

[MIT license](https://github.com/acnazarejr/fdet/blob/master/LICENSE)
