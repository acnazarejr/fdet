"""Detector base class"""

from abc import ABC, abstractmethod
from typing import Optional, List, Union, Tuple, Dict, Sequence, Iterable#, Hashable
import torch
import numpy as np
import cv2


InType = Union[np.ndarray, str]

ConfType = float
BoxType = Tuple[int, int, int, int]
KeypointsType = Dict[str, Tuple[int, int]]
SingleDetType = Dict[str, Union[ConfType, BoxType, KeypointsType]]
OutType = List[SingleDetType]

# IterableInType = Union[Tuple[Hashable, np.ndarray], Sequence[Tuple[Hashable, np.ndarray]]]


class Detector(ABC):
    """Abstract base class for Detectors"""


    def __init__(self, cuda_devices: Optional[List[int]] = None,
                 cuda_enable: bool = torch.cuda.is_available()) -> None:

        self._cuda_enable = cuda_enable and torch.cuda.is_available()
        if self._cuda_enable:
            self._cuda_devices = _get_torch_devices(cuda_devices)
        else:
            self._cuda_devices = list()

        self._cuda_enable = self._cuda_enable and bool(self._cuda_devices)
        self._device_control = self._cuda_devices[0] if self._cuda_enable else torch.device('cpu')


    @property
    def cuda_enable(self) -> bool:
        """cudda_enable"""
        return self._cuda_enable

    @property
    def cuda_devices(self) -> List[torch.device]:

        """cuda_devices"""
        return self._cuda_devices


    def detect(self, data: Union[InType, Iterable[InType]]) -> Union[OutType, List[OutType]]:
        """detect"""

        np_data = self.__check_input_data(data)
        n_images, _, _, _ = np_data.shape

        batch_detections = self._detect_batch(np_data)

        torch.cuda.empty_cache()

        if n_images == 1:
            return batch_detections[0]
        return batch_detections

    # def video_detect(self, video: vop.VideoHandle, batch_size: int = None,
    #                  show_progress: bool = True, leave_progress: bool = True) -> List[DetType]:
    #     """video_detect"""
    #     if batch_size is None:
    #         batch_size = 1

    #     if batch_size <= 0:
    #         raise ValueError(f'Invalid batch size {batch_size}')

    #     video.reset()
    #     self.stats.clear()
    #     pbar = tqdm(video, disable=not show_progress, leave=leave_progress, total=len(video))

    #     batch_images: List[np.ndarray] = list()
    #     all_detections: List[DetType] = list()

    #     def __run_batch(batch):
    #         batch_detections = self.detect(batch)
    #         if batch_size == 1:
    #             all_detections.append(batch_detections) #type: ignore
    #         else:
    #             all_detections.extend(batch_detections) #type: ignore

    #     for frame in pbar:
    #         batch_images.append(frame)
    #         if len(batch_images) >= batch_size:
    #             __run_batch(batch_images)
    #             batch_images.clear()
    #         pbar.set_postfix(fps=self.stats.fps, last_fps=self.stats.last_fps,
    #                          average=self.stats.average_image_time)
    #     if batch_images:
    #         __run_batch(batch_images)
    #         pbar.set_postfix(fps=self.stats.fps, last_fps=self.stats.last_fps,
    #                          average=self.stats.average_image_time)
    #     pbar.update()
    #     return all_detections


    # def iterative_detect(self, iterable: Iterable[IterableInType], show_progress: bool = True,
    #                      leave_progress: bool = True) -> Dict[Hashable, DetType]:
    #     """iterative_detect"""

    #     self.stats.clear()
    #     pbar = tqdm(iterable, disable=not show_progress, leave=leave_progress)

    #     all_detections: Dict = OrderedDict()
    #     for data in pbar:
    #         batch = data if isinstance(data, list) else [data]
    #         batch_keys, batch_images = zip(*batch)
    #         batch_detections = self.detect(batch_images)
    #         if len(batch_keys) == 1:
    #             all_detections[batch_keys[0]] = batch_detections
    #         else:
    #             all_detections.update(zip(batch_keys, batch_detections))
    #         pbar.set_postfix(
    #             fps=self.stats.fps,
    #             last_fps=self.stats.last_fps,
    #             average=self.stats.average_image_time)

    #     return all_detections


    @abstractmethod
    def _detect_batch(self, data: List[np.ndarray]) -> List[OutType]:
        raise NotImplementedError('Abstract method!')

    @staticmethod
    def __check_input_data(data: Union[InType, Sequence[InType]]) -> np.ndarray:

        def __read_as_rgb(path):
            return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

        if isinstance(data, str):
            data = __read_as_rgb(data)

        if isinstance(data, Sequence):
            data = [
                image if isinstance(image, np.ndarray) else __read_as_rgb(data) for image in data
            ]
            data = np.stack(data)

        if isinstance(data, np.ndarray):
            np_data = data
            if np_data.ndim == 3:
                np_data = np.expand_dims(data, axis=0)
            if np_data.ndim != 4:
                raise ValueError(f'Invalid data shape {np_data.shape}')
            return np_data

        raise ValueError(f'Invalid input data type {type(data)}')

    def _init_torch_module(self, module: torch.nn.Module) -> torch.nn.Module:
        if self._cuda_enable:
            if len(self._cuda_devices) == 1:
                return _init_single_gpu_module(module, self._device_control)
            return _init_multi_gpu_module(module, self._cuda_devices)
        return module.to(self._device_control)


def _get_torch_devices(devices_index: Optional[List[int]] = None) -> List[torch.device]:

    def __create_cuda_device(index: int) -> torch.device:
        if not torch.cuda.is_available():
            raise EnvironmentError('Cuda is not available in this host.')
        if index not in list(range(torch.cuda.device_count())):
            raise ValueError('Invalid device index: '+ str(index))
        return torch.device('cuda', index)

    if devices_index is None:
        devices_index = list(range(torch.cuda.device_count()))
    return [__create_cuda_device(index) for index in devices_index]

def _init_single_gpu_module(module: torch.nn.Module, torch_device: torch.device) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise EnvironmentError('Cuda is not available in this host.')
    if torch_device.type != 'cuda':
        raise ValueError('Invalid device type: ' + str(torch_device.type))
    if torch_device.index not in list(range(torch.cuda.device_count())):
        raise ValueError('Invalid device index: '  + str(torch_device.index))
    return module.to(torch_device)

def _init_multi_gpu_module(module: torch.nn.Module,
                           torch_devices: List[torch.device]) -> torch.nn.Module:
    if not torch.cuda.is_available():
        raise EnvironmentError('Cuda is not available in this host.')

    if len(torch_devices) < 2:
        raise ValueError('A multi-gpu module requires at least two cuda devices.')

    for torch_device in torch_devices:
        if torch_device.type != 'cuda':
            raise ValueError('Invalid device type: ' + str(torch_device.type))
        if torch_device.index not in list(range(torch.cuda.device_count())):
            raise ValueError('Invalid device index: '  + str(torch_device.index))

    module = torch.nn.DataParallel(module, device_ids=torch_devices)
    module = module.to(torch_devices[0])
    return module
