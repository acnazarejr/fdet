"""cli entry-points"""

import os
import json
from typing import Union, Tuple, Iterable, Hashable, Dict
import functools
import numpy as np
import cv2
import click
from tqdm import tqdm
from click_option_group import RequiredMutuallyExclusiveOptionGroup as OptionGroup
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
            default=None, help='Specify the GPU device', type=int, show_default=True,
        ),
        click.option(
            '-o', '--output', 'output_file',
            required=True, default='detections.json', show_default=True,
            help='The path of the output json file',
            type=click.Path(exists=False, file_okay=True, dir_okay=False, writable=True,
                            readable=False, resolve_path=True, allow_dash=False, path_type=None)
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
    '-m', '--min-size', 'min_size', type=int, default=20, show_default=True,
    help='Minimum size of face to detect, in pixels'
)
@click.option(
    '-t', '--thresholds', 'thresholds',
    default=(0.6, 0.7, 0.8), show_default=True, nargs=3, type=float,
    help='The thresholds fo each MTCNN step'
)
@click.option(
    '-n', '--nms', 'nms',
    default=(0.7, 0.7, 0.7), show_default=True, nargs=3, type=float,
    help='The NMS thresholds fo each MTCNN step'
)
def mtcnn(**kwargs):
    """mtcnn detector"""

    input_data = _process_input(kwargs)

    cuda_enable = kwargs.get('cuda')
    cuda_devices = [kwargs.get('gpu')] if kwargs.get('gpu') is not None else None
    min_face_size = kwargs.get('min_size')
    thresholds = kwargs.get('thresholds')
    nms_thresholds = kwargs.get('nms')

    detector = fdet.MTCNN(min_face_size=min_face_size, thresholds=thresholds,
                          nms_thresholds=nms_thresholds, cuda_devices=cuda_devices,
                          cuda_enable=cuda_enable)

    detections = _detect(detector, input_data, kwargs.get('save_frames_dir'), kwargs.get('quiet'))

    with open(kwargs.get('output_file'), 'w') as pfile:
        json.dump(detections, pfile)

@main.command()
@_common_options
@click.option(
    '-b', '--backbone', 'backbone',
    type=click.Choice(['RESNET50', 'MOBILENET'], case_sensitive=False),
    required=True, help='The backbone network'
)
@click.option(
    '-m', '--max-size', 'max_size', type=int, default=1000, show_default=True,
    help='Maximun size of face to detect, in pixels'
)
@click.option(
    '-t', '--threshold', 'threshold',
    default=0.8, show_default=True, type=float,
    help='The confidence threshold'
)
@click.option(
    '-n', '--nms', 'nms',
    default=0.4, show_default=True, type=float,
    help='The NMS threshold'
)
def retinaface(**kwargs):
    """retinaface detector"""
    input_data = _process_input(kwargs)

    cuda_enable = kwargs.get('cuda')
    cuda_devices = [kwargs.get('gpu')] if kwargs.get('gpu') is not None else None
    backbone = kwargs.get('backbone')
    max_face_size = kwargs.get('max_size')
    threshold = kwargs.get('threshold')
    nms_threshold = kwargs.get('nms')

    detector = fdet.RetinaFace(backbone=backbone, max_face_size=max_face_size, threshold=threshold,
                               nms_threshold=nms_threshold, cuda_devices=cuda_devices,
                               cuda_enable=cuda_enable)

    detections = _detect(detector, input_data, kwargs.get('save_frames_dir'), kwargs.get('quiet'))

    with open(kwargs.get('output_file'), 'w') as pfile:
        json.dump(detections, pfile)


def _detect(detector, input_data, save_frames_dir, quiet) -> Dict:

    if save_frames_dir is not None and not os.path.exists(save_frames_dir):
        os.makedirs(save_frames_dir)

    detections = dict()
    for key, image in tqdm(input_data, disable=quiet, leave=True):
        detections[key] = detector.detect(image)
        if save_frames_dir is not None:
            image_output = fdet.io.draw_detections(image, detections[key], color='blue')
            filename = key if isinstance(key, str) else str(key) + '.png'
            fdet.io.save(os.path.join(save_frames_dir, filename), image_output)
    return detections

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
