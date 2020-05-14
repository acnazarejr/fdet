"""cli entry-points"""

import os
import json
from typing import Union, Tuple, Iterable, Hashable, Dict, Any
import collections
import functools
import numpy as np
import torch
import click
from tqdm import tqdm
from fdet.detector import Detector
import fdet

VALID_IMG_EXTENSTIONS = (
    '.bmp',
    '.pbm', '.pgm', '.ppm',
    '.sr', '.ras',
    '.jpeg', '.jpg', '.jpe',
    '.jp2',
    '.tiff', '.tif',
    '.png'
)


@click.group()
@click.version_option(fdet.__version__)
def main():
    """main fdet cli function"""

def _common_options(func):
    options = [
        click.option(
            '-i', '--image', 'image_file', multiple=True,
            help='The path of the image to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None),
            callback=(lambda ctx, param, value: value if value else None)
        ),
        click.option(
            '-v', '--video', 'video_file',
            help='The path of the video input to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        click.option(
            '-l', '--list', 'images_list',
            help='The path of a text file containing a list of images to detect',
            type=click.Path(exists=True, file_okay=True, dir_okay=False, writable=False,
                            readable=True, resolve_path=True, allow_dash=False, path_type=None)
        ),
        click.option(
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
            '-p', '--print', is_flag=True, show_default=True, help='Enables the print output mode',
        ),
        click.option(
            '-bs', '--batch-size', 'batch_size',
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
            '--color', 'draw_color',
            required=False, default='blue', show_default=True,
            help='The color of detected faces drawn in output frames.',
            type=str
        ),
        click.option(
            '-q', '--quiet', is_flag=True, show_default=True, help='Enables quiet mode'
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
    type=click.Choice(fdet.RetinaFace.valid_backbones(), case_sensitive=False),
    required=True, help='The backbone network'
)
@click.option(
    '-m', '--max-size', 'max_size',
    type=click.IntRange(100, 1000),
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
                image_output = fdet.io.draw_detections(img, image_detections,
                                                       color=kwargs.get('draw_color'),
                                                       thickness=3)
                filename = image_key if isinstance(image_key, str) else str(image_key) + '.png'
                filename = os.path.join(str(kwargs.get('save_frames_dir')), filename)
                fdet.io.save(filename, image_output)
        return batch_response


    if kwargs.get('save_frames_dir') is not None:
        os.makedirs(str(kwargs.get('save_frames_dir')), exist_ok=True)

    detections = collections.OrderedDict()
    batch = list()
    pbar = tqdm(input_data, disable=kwargs.get('quiet') or len(input_data) == 1, leave=True)
    for key, image in pbar:

        if isinstance(image, str):
            image = fdet.io.read_as_rgb(image)

        batch.append((key, image))
        if len(batch) == int(kwargs.get('batch_size', 1)):
            detections.update(__process_batch(batch))
            batch.clear()

    if batch:
        detections.update(__process_batch(batch))

    if kwargs.get('print'):
        for image_key, image_detections in detections.items():
            filename = image_key if isinstance(image_key, str) else 'frame ' + str(image_key)
            msg = click.style('Detected faces on {}: '.format(filename),
                              fg='yellow', bold=True, reset=False,)
            msg += click.style(' {:04d} '.format(len(image_detections)),
                               fg='yellow', bold=True, reset=True)
            click.secho('=' * 118, fg='yellow', bold=True, reset=True)
            click.echo(msg)
            click.secho('-' * 118, fg='yellow', bold=True, reset=True)
            colors = ['bright_magenta', 'bright_green', 'cyan']
            for idx, detection in enumerate(image_detections):

                msg_box = 'BBox: ({:4d} {:4d} {:4d} {:4d})'.format(*detection['box'])
                msg_conf = 'Confidence: {:2.4f}'.format(detection['confidence'])
                msg_keys = '         Keypoints: LEyes ({:4d} {:4d}) | REyes ({:4d} {:4d}) | '
                msg_keys += 'Nose ({:4d} {:4d}) | LMouth ({:4d} {:4d}) | RMouth ({:4d} {:4d})'
                kpts = detection['keypoints']
                msg_keys = msg_keys.format(*kpts['left_eye'], *kpts['right_eye'], *kpts['nose'],
                                           *kpts['mouth_left'], *kpts['mouth_right'])
                msg = '  {:04d} - {} | {}\n{}'.format(idx+1, msg_box, msg_conf, msg_keys)
                click.echo(click.style(msg, fg=colors[idx%3]))
            click.secho('=' * 118, fg='yellow', bold=True, reset=True)
            click.echo()

    os.makedirs(os.path.dirname(str(kwargs.get('output_file'))), exist_ok=True)
    with open(str(kwargs.get('output_file')), 'w') as pfile:
        json.dump(detections, pfile)


def _process_input(kwargs) -> Iterable[Tuple[Hashable, Union[str, np.ndarray]]]:

    input_options = [kwargs.get('image_file'), kwargs.get('video_file'),
                     kwargs.get('images_list'), kwargs.get('images_dir')]
    not_none_sum = sum(1 for _ in filter(None.__ne__, input_options))

    if not_none_sum == 0:
        error_msg = 'Missing one of the required options from input data sources:\n'
        error_msg += "'-d' / '--dir'\n"
        error_msg += "'-l' / '--list'\n"
        error_msg += "'-v' / '--video'\n"
        error_msg += "'-i' / '--image'\n"
        raise click.UsageError(error_msg)

    if not_none_sum > 1:
        error_msg = 'The given mutually exclusive options cannot be used at the same time:\n'
        error_msg += "'-d' / '--dir'\n"
        error_msg += "'-l' / '--list'\n"
        error_msg += "'-v' / '--video'\n"
        error_msg += "'-i' / '--image'\n"
        raise click.UsageError(error_msg)

    if kwargs.get('image_file') is not None and kwargs.get('image_file'):
        return [(os.path.basename(ifile), ifile) for ifile in kwargs.get('image_file')]
    if kwargs.get('video_file') is not None:
        return fdet.io.VideoHandle(kwargs.get('video_file'))
    if kwargs.get('images_list') is not None:
        return [
            (os.path.basename(ifile), ifile)
            for ifile in open(kwargs.get('images_list'), 'r').read().splitlines()
        ]

    #kwargs.get('images_dir') is not None:
    list_of_valid_files = list()
    for ifile in os.listdir(kwargs.get('images_dir')):
        image_path = os.path.join(kwargs.get('images_dir'), ifile)
        if os.path.splitext(image_path)[1] in VALID_IMG_EXTENSTIONS:
            list_of_valid_files.append((ifile, image_path))
    return list_of_valid_files
