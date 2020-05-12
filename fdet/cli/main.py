"""cli entry-points"""

import os
import json
from typing import Union, Tuple, Iterable, Hashable, Dict, Any
import collections
import functools
import numpy as np
import torch
import cv2
import click
from tqdm import tqdm
from click_option_group import RequiredMutuallyExclusiveOptionGroup as OptionGroup
from fdet.detector import Detector
import fdet

@click.group()
def main():
    """main fdet cli function"""

def _common_options(func):
    option_group = OptionGroup('Input data sources', help='The sources of the input data')
    options = [
        option_group.option(
            '-i', '--image', 'image_file', multiple=True,
            help='The path of the image to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        option_group.option(
            '-v', '--video', 'video_file',
            help='The path of the video input to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        option_group.option(
            '-l', '--list', 'images_list',
            help='The path of a text file containing a list of images to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        option_group.option(
            '-d', '--dir', 'images_dir',
            help='The path of a folder containing the images to detect',
            type=click.Path(exists=True, file_okay=False, dir_okay=True, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        click.option(
            '--cuda/--no-cuda',
            default=True, help='Enables CUDA utilization', show_default=True,
        ),
        click.option(
            '-g', '--gpu', 'gpu',
            type=click.IntRange(0, torch.cuda.device_count() - 1),
            default=None, help='Specify the GPU device', show_default=True,
            callback=(lambda ctx, param, value: [value] if value is not None else None)
        ),
        click.option(
            '-o', '--output', 'output_file',
            required=True, default='detections.json', show_default=True,
            help='The path of the output json file',
            type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True,
                            readable=False, resolve_path=True, allow_dash=False, path_type=None)
        ),
        click.option(
            '-b', '--batch-size', 'batch_size',
            required=False,
            help='The size of the detect batch (useful for multiple inputs of the same size)',
            type=click.IntRange(0, 1000), default=1, show_default=True,
        ),
        click.option(
            '-s', '--save-frames', 'save_frames_dir',
            required=False,
            help='The path of the directory to save output frames, with detections',
            type=click.Path(exists=False, file_okay=False, dir_okay=True, writable=True,
                            readable=False, resolve_path=True, allow_dash=False, path_type=None)
        ),
        click.option(
            '--quiet', is_flag=True, show_default=True, help='Enables quiet mode'
        )
    ]
    return functools.reduce(lambda x, opt: opt(x), options, func)


@main.command()
@_common_options
@click.option(
    '-m', '--min-size', 'min_size',
    type=click.IntRange(20, 1000),
    default=20, show_default=True,
    help='Minimum size of face to detect, in pixels'
)
@click.option(
    '-t', '--thresholds', 'thresholds',
    default=(0.6, 0.7, 0.8), show_default=True, nargs=3, type=click.FloatRange(0, 1),
    help='The thresholds fo each MTCNN step'
)
@click.option(
    '-n', '--nms', 'nms',
    default=(0.7, 0.7, 0.7), show_default=True, nargs=3, type=click.FloatRange(0, 1),
    help='The NMS thresholds fo each MTCNN step'
)
def mtcnn(**kwargs):
    """mtcnn detector"""

    input_data = _process_input(kwargs)
    detector = fdet.MTCNN(min_face_size=kwargs.get('min_size'), thresholds=kwargs.get('thresholds'),
                          nms_thresholds=kwargs.get('nms'), cuda_devices=kwargs.get('gpu'),
                          cuda_enable=kwargs.get('cuda'))

    _detect(detector, input_data, kwargs)

@main.command()
@_common_options
@click.option(
    '-b', '--backbone', 'backbone',
    type=click.Choice(['RESNET50', 'MOBILENET'], case_sensitive=False),
    required=True, help='The backbone network'
)
@click.option(
    '-m', '--max-size', 'max_size',
    type=click.IntRange(20, 1000),
    default=1000, show_default=True,
    help='Maximun size of face to detect, in pixels'
)
@click.option(
    '-t', '--threshold', 'threshold',
    default=0.8, show_default=True, type=click.FloatRange(0, 1),
    help='The confidence threshold'
)
@click.option(
    '-n', '--nms', 'nms',
    default=0.4, show_default=True, type=click.FloatRange(0, 1),
    help='The NMS threshold'
)
def retinaface(**kwargs):
    """retinaface detector"""
    input_data = _process_input(kwargs)

    detector = fdet.RetinaFace(backbone=kwargs.get('backbone'),
                               max_face_size=kwargs.get('max_size'),
                               threshold=kwargs.get('threshold'),
                               nms_threshold=kwargs.get('nms'), cuda_devices=kwargs.get('gpu'),
                               cuda_enable=kwargs.get('cuda'))

    _detect(detector, input_data, kwargs)


def _detect(detector: Detector, input_data, kwargs: Dict[str, Any]) -> None:


    def __process_batch(current_batch: Iterable[Tuple[Hashable, np.ndarray]]) -> Dict:

        batch_keys, batch_images = zip(*current_batch)
        batch_detections = detector.batch_detect(batch_images)

        batch_response = collections.OrderedDict()
        for image_key, image_detections, img in zip(batch_keys, batch_detections, batch_images):
            batch_response[image_key] = image_detections
            if kwargs.get('save_frames_dir') is not None:
                image_output = fdet.io.draw_detections(img, image_detections, color='blue')
                filename = image_key if isinstance(image_key, str) else str(image_key) + '.png'
                filename = os.path.join(str(kwargs.get('save_frames_dir')), filename)
                fdet.io.save(filename, image_output)
        return batch_response


    if kwargs.get('save_frames_dir') is not None:
        os.makedirs(str(kwargs.get('save_frames_dir')), exist_ok=True)

    detections = collections.OrderedDict()
    batch = list()
    for key, image in tqdm(input_data, disable=kwargs.get('quiet'), leave=True):

        if isinstance(image, str):
            image = fdet.io.read_as_rgb(image)

        batch.append((key, image))
        if len(batch) == int(kwargs.get('batch_size', 1)):
            detections.update(__process_batch(batch))
            batch.clear()

    if batch:
        detections.update(__process_batch(batch))

    with open(str(kwargs.get('output_file')), 'w') as pfile:
        json.dump(detections, pfile)


def _process_input(kwargs) -> Iterable[Tuple[Hashable, Union[str, np.ndarray]]]:
    try:
        if kwargs['image_file'] is not None and kwargs['image_file']:
            return [(os.path.basename(ifile), ifile) for ifile in kwargs['image_file']]
        if kwargs['video_file'] is not None:
            return fdet.io.VideoHandle(kwargs['video_file'])
        if kwargs['images_list'] is not None:
            return [
                (os.path.basename(ifile), ifile)
                for ifile in open(kwargs['images_list'], 'r').read().splitlines()
            ]
        if kwargs['images_dir'] is not None:
            return [
                (ifile, os.path.join(kwargs['images_dir'], ifile))
                for ifile in os.listdir(kwargs['images_dir'])
            ]
        raise IOError('Invalid Input type')
    except cv2.error as cv_error:
        raise click.ClickException(str(cv_error))
    except IOError as ioerror:
        raise click.ClickException(str(ioerror))
    except Exception as error:
        raise click.ClickException('Unexpected error: ' + str(error))
