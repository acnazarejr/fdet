# FDet - Deep Learning Face Detection

[![Build](https://github.com/acnazarejr/fdet/workflows/Build/badge.svg)](https://github.com/acnazarejr/fdet/actions)
[![CodeFactor](https://www.codefactor.io/repository/github/acnazarejr/fdet/badge)](https://www.codefactor.io/repository/github/acnazarejr/fdet)
[![codecov](https://codecov.io/gh/acnazarejr/fdet/branch/master/graph/badge.svg)](https://codecov.io/gh/acnazarejr/fdet)
[![Platform](https://img.shields.io/badge/os-linux%20%7C%20win%20%7C%20mac-blue)](https://pypi.org/project/fdet/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/fdet)](https://pypi.org/project/fdet/)
[![PyPI](https://img.shields.io/pypi/v/fdet)](https://pypi.org/project/fdet/)
[![GitHub release (latest by date)](https://img.shields.io/github/v/release/acnazarejr/fdet)](https://github.com/acnazarejr/fdet/releases)
[![GitHub](https://img.shields.io/github/license/acnazarejr/fdet)](https://github.com/acnazarejr/fdet/blob/master/LICENSE)

The `fdet` is a ready-to-use implementation of deep learning face detectors with landkmarks.

![Example](https://github.com/acnazarejr/fdet/raw/master/assets/example.jpg)

You can use it directly in your code, as a [python library](#python-library-usage):

```python
>>> from fdet import io, RetinaFace

>>> detector = RetinaFace(backbone='RESNET50')
>>> image = io.read_as_rgb('path_to_image.jpg')
>>> detector.detect(image)
[{'box': [511, 47, 35, 45], 'confidence': 0.9999996423721313,
  'keypoints': {'left_eye': [517, 70], 'right_eye': [530, 65], 'nose': [520, 77],
                'mouth_left': [522, 87], 'mouth_right': [531, 83]}}]
```

Or through [command-line](#command-line-usage) application:

```bash
fdet retinaface -b RESNET50 -i path_to_image.jpg -o detections.json --gpu 1
```

## **Features**

Currently, there are two different detectors available on FDet:

- **MTCNN** - Joint face detection and alignment using multitask cascaded convolutional networks [[zhang:2016]](#references)
- **RetinaFace** - Single-stage dense face localisation in the wild. [[deng:2019]](#references). You can use it with two different backbones:
  - *MobileNet*: Fast and light-weighted model (achieves high FPS)
  - *Resnet50*: A medium-size model for better results, but slower.

Despite the availability of different implementations of these algorithms, there are some disadvantages we found when using them. So we create this project to offer the following features, in one package:

- :star: Real-time face detection;
- :star: Support for batch detection (useful for fast detection in multiple images and videos);
- :star: Ease of use through python library or command-line tool;
- :star: Provide a unified interface to assign 'CPU' or 'GPU' devices;
- :star: Multiple GPU's support;
- :star: Automatic on-demand model weights download;
- :star: Compatible with Windows, Linux, and macOS systems.

## **Installation**

1. You need to [install PyTorch](https://pytorch.org/get-started/locally/) first (if you have a GPU, install PyTorch with CUDA support).

2. Then `fdet` can be installed through pip:

```bash
pip install fdet
```

## **Command-line Usage**

Simply and fast usage through command-line tool.

The **`fdet`** command-line tool has two sub-commands, on for each available detector: **`fdet mtcnn`** and **`fdet retinaface`**.

![Terminal](https://github.com/acnazarejr/fdet/raw/master/assets/terminal.gif)

### **Options**

For a detailed list of available options type: **`fdet mtcnn --help`** or **`fdet retinaface --help`**, according to the desired detector.

#### Data Input

> This options are mutually exclusive

- `-i, --image FILE`: Image to detect. You can specify multiple images (`-i img1.jpg -i img2.jpg`)
- `-v, --video FILE`: Video file to detect. Only one video can be specified at a time.
- `-l, --list FILE`: Text file containing a list of images (absolute paths) to detect.
- `-d, --dir DIRECTORY`: The path of a directory containing images to detect. Ignores files that are not images.

#### Data Output

- `-o, --output FILE`: Path to the output json file containing the detections.
- `-s, --save-frimes DIRECTORY` *(Optional)*: If specified, folder to save the output images with the detected faces drawn. Be careful when using this option with video input, as it will save all frames of the video.
- `p, --print` *(Optional)*: If specified,, print the detections to the console screen.
- `-q, --quiet` *(Optional)*: Do not display progress bar or any results.

#### Execution

- `--no-cuda`: Disables the CUDA utilization. When CUDA is not supported, it is automatically disabled.
- `-g, --GPU INT` *(Optional)*: When CUDA is supported, specifies which GPU to use. If not set, all available GPUs will be used.
- `-bs, --batch-size INT` *(Optional)*: The size of the detection batch (providing considerable speed-up) [default: 1]. **This option only works for multiple images when they are the same size**.

> Defining the batch size is a complex task because it depends on the available memory in the system. We recommend performing small preliminary tests to find a suitable value.

## **Python Library Usage**

If you want to use `fdet` from python, just import it,

```python
from  fdet import MTCNN, RetinaFace
```

and instantiate your desired detector, with its respective parameters:

- **`MTCNN(thresholds, nms_thresholds, min_face_size, cuda_enable, cuda_devices)`**
  - `thresholds` (tuple, optional): The thresholds fo each MTCNN step [default: (0.6, 0.7, 0.8)]
  - `nms_thresholds` (tuple, optional): The NMS thresholds fo each MTCNN step [default: (0.7, 0.7, 0.7)]
  - `min_face_size` (float, optional): The minimum size of the face to detect, in pixels [default: 20.0].
  - `cuda_enable` (bool, optional): Indicates wheter CUDA, if available, should be used or not. If False, uses only CPU processing [default: True].
  - `cuda_devices` (list, optional): List of CUDA GPUs to be used. If None, uses all avaliable GPUs [default: None]. If `cuda_enable` is False, this parameter is ignored.

- **`RetinaFace(backbone, threshold, nms_threshold, max_face_size, cuda_enable, cuda_devices)`**
  - `backbone` (str): The backbone model [`'RESNET50'` or `'MOBILENET'`].
  - `threshold` (tuple, optional): The detection threshold [default: 0.8]
  - `nms_threshold` (tuple, optional): The NMS threshold [default: 0.4]
  - `max_face_size` (int, optional): The maximum size of the face to detect, in pixels [default: 1000].
  - `cuda_enable` (bool, optional): Indicates wheter CUDA, if available, should be used or not. If False, uses only CPU processing. [default: True].
  - `cuda_devices` (list, optional): List of CUDA GPUs to be used. If None, uses all avaliable GPUs. [default: None]. If `cuda_enable` is False, this parameter is ignored.

To perform detection you can simply use the following methods provided by the classes:

**`detect(image: np.ndarray)`**: Single-image detection ([example](#singe-image-detection-example)).

**`batch_detect(image: np.ndarray)`**: Performs face detection on image batches, typically providing considerable speed-up ([example](#batch-detection-example)).

For each processed image, the detector returns a list of `dict` objects, which in turn represent the detected faces. The `dict` contains three main keys, described below.

```python
[
  {'box': [511, 47, 35, 45], 'confidence': 0.9999996423721313,
  'keypoints': {'left_eye': [517, 70], 'right_eye': [530, 65], 'nose': [520, 77],
                'mouth_left': [522, 87], 'mouth_right': [531, 83]}}
]
```

- `'box'`: The bounding box formatted as a list `[x, y, width, height]`;
- `'confidence'`: The probability for a bounding box to be matching a face;
- `'keypoints'`: The five landmarks formatted into a `dict` with the keys `'left_eye'`, `'right_eye'`, `'nose'`, `'mouth_left'`, `'mouth_right'`. Each keypoint is identified by a pixel position `[x, y]`.

> The `batch_detect()` method will return a `list` of `lists` containing the results of all batch images.

### **Singe-Image Detection Example**

This example shows how to detect faces, using a single image, and draw the detections in an output image.

```python
>>> from fdet import io, MTCNN
>>>
>>> detector = MTCNN()
>>>
>>> image = io.read_as_rgb('example.jpg')
>>> detections = detector.detect(image)
>>>
>>> output_image = io.draw_detections(image, detections, color='white', thickness=5)
>>> io.save('output.jpg', output_image)
```

> The `io.read_as_rgb()` is a wrapper for opencv `cv2.imread()` to ensure an RGB image and can be replaced by:
>
> ```python
> image = cv2.imread('example.jpg', cv2.IMREAD_COLOR)
> image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
> ```

### **Batch Detection Example**

A batch should be structured as list of images (`numpy` arrays)  of equal dimension. The returned detections list will have an additional first dimension corresponding to the batch size. Each image in the batch may have one or more faces detected.

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
