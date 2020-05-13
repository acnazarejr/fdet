"""Detector base class"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Dict, Sequence
import collections.abc
import torch
import numpy as np
from fdet.utils.errors import DetectorCudaError, DetectorValueError, DetectorInputError


BoxType = List[int]
KeypointsType = Dict[str, List[int]]
SingleDetType = Dict[str, Union[float, BoxType, KeypointsType]]
ImageDetOutType = List[SingleDetType]
BatchDetOutType = List[ImageDetOutType]


class Detector(ABC):
    """Abstract base class for Detectors"""


    def __init__(self, cuda_enable: bool = torch.cuda.is_available(),
                 cuda_devices: Optional[Sequence[int]] = None, cuda_benchmark: bool = True) -> None:


        if not isinstance(cuda_enable, bool):
            raise DetectorValueError('The cuda_enable value must be a boolean.')
        self._cuda_enable = cuda_enable and torch.cuda.is_available()

        if cuda_devices is not None and not isinstance(cuda_devices, collections.abc.Sequence):
            raise DetectorValueError('The cuda_devices value must be a sequence.')

        if self._cuda_enable:
            self._cuda_devices = _get_torch_devices(cuda_devices)
        else:
            self._cuda_devices = list()

        self._cuda_enable = self._cuda_enable and bool(self._cuda_devices)
        self._device_control = self._cuda_devices[0] if self._cuda_enable else torch.device('cpu')

        if not isinstance(cuda_benchmark, bool):
            raise DetectorValueError('The cuda_benchmark value must be a boolean.')
        self._cuda_benchmark = cuda_benchmark


    @abstractmethod
    def _run_data_batch(self, data: np.ndarray) -> BatchDetOutType:
        raise NotImplementedError('Abstract method!')

    def detect(self, image: np.ndarray) -> ImageDetOutType:
        """
        Detects bounding boxes, and with respective landmark points, from the specified image.

        Args:
            image (np.ndarray): The input image to process.

        Returns:
            List[Dict]: A list containing all the bounding boxes and lankdmarks detected.
        """
        np_data = self.__check_image(image)
        batch_detections = self._run_data_batch(np_data)
        return batch_detections[0]

    def batch_detect(self, image_batch: Union[np.ndarray, Sequence[np.ndarray]]) -> BatchDetOutType:
        """
        Detects bounding boxes, and with respective landmark points, from the specified images in
        the bach. The batch size must be defined by the user.

        Args:
            image_batch (Sequence[np.ndarray]): The images sequence to process.

        Returns:
            List[List[Det]]: A list of lists. For each image in the batch, returns a list of all the
                bounding boxes and lankdmarks detected in the respective image.
        """
        np_data = self.__check_batch(image_batch)
        return self._run_data_batch(np_data)


    @staticmethod
    def __check_image(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray):
            raise DetectorInputError('The input image must be numpy array.')
        if not image.ndim == 3:
            raise DetectorInputError(
                'The input image must have 3 dimensions (W x H X channels). Got: ' + str(image.ndim)
            )
        if not image.shape[2] == 3:
            raise DetectorInputError(
                'The input image must have 3 channels (R, G and B). Got: ' + str(image.shape[2])
            )
        return np.expand_dims(image, axis=0)

    @staticmethod
    def __check_batch(batch: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:

        if not isinstance(batch, (np.ndarray, collections.abc.Sequence)):
            raise DetectorInputError('The batch must be a Sequence or numpy array.')

        if isinstance(batch, collections.abc.Sequence):
            if not batch:
                raise DetectorInputError('The batch must have least one image.')
            batch = np.stack(batch)


        if isinstance(batch, np.ndarray) and not batch.ndim == 4:
            raise DetectorInputError(
                'All images must have 3 dimensions (W x H X channels). Got: ' + str(batch.ndim - 1)
            )

        if isinstance(batch, np.ndarray) and not batch.shape[3] == 3:
            raise DetectorInputError(
                'All images must have 3 channels (R, G and B). Got: ' + str(batch.shape[3])
            )

        return batch

    def _init_torch_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if self._cuda_enable:
            if len(self._cuda_devices) == 1:
                return _init_single_gpu_module(module, self._device_control)
            return _init_multi_gpu_module(module, self._cuda_devices)
        return module.to(self._device_control)


def _get_torch_devices(devices_index: Optional[Sequence[int]] = None) -> List[torch.device]:

    def __create_cuda_device(index: int) -> torch.device:
        if not torch.cuda.is_available():
            raise DetectorCudaError('CUDA is not available in this host.')
        if index not in list(range(torch.cuda.device_count())):
            raise DetectorCudaError('Invalid CUDA device index: '+ str(index))
        return torch.device('cuda', index)

    if devices_index is None:
        devices_index = list(range(torch.cuda.device_count()))
    return [__create_cuda_device(index) for index in devices_index]

def _init_single_gpu_module(module: torch.nn.Module, torch_device: torch.device) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise DetectorCudaError('CUDA is not available in this host.')
    if torch_device.type != 'cuda':
        raise DetectorCudaError('Invalid torch CUDA device type: ' + str(torch_device.type))
    if torch_device.index not in list(range(torch.cuda.device_count())):
        raise DetectorCudaError('Invalid torch CUDA device index: '  + str(torch_device.index))
    return module.to(torch_device)

def _init_multi_gpu_module(module: torch.nn.Module,
                           torch_devices: List[torch.device]) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise DetectorCudaError('CUDA is not available in this host.')

    if len(torch_devices) < 2:
        raise DetectorCudaError('A multi-gpu module requires at least two CUDA devices.')

    for torch_device in torch_devices:
        if torch_device.type != 'cuda':
            raise DetectorCudaError('Invalid torch CUDA device type: ' + str(torch_device.type))
        if torch_device.index not in list(range(torch.cuda.device_count())):
            raise DetectorCudaError('Invalid torch CUDA device index: '  + str(torch_device.index))

    module = torch.nn.DataParallel(module, device_ids=torch_devices)
    module = module.to(torch_devices[0])
    return module
