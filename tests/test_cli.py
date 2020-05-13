"""Tests over cli"""

#pylint: skip-file

import os
import tempfile
import pytest
from click.testing import CliRunner
from fdet.cli.main import main

@pytest.fixture
def resources_path():
    return os.path.join(os.path.dirname(__file__), 'resources')

@pytest.fixture
def image_path(resources_path):
    return os.path.join(resources_path, 'images', 'low.jpg')

@pytest.fixture
def video_path(resources_path):
    return os.path.join(resources_path, 'video.mp4')

@pytest.fixture
def frames_path(resources_path):
    return os.path.join(resources_path, 'frames')


def test_cli_invalid(image_path, video_path, frames_path):
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as temp_dir:
        output = os.path.join(temp_dir, 'out.json')
        result = runner.invoke(main, ['mtcnn', '-i', image_path, '-v', video_path, '-o', output])
        assert result.exit_code != 0
        result = runner.invoke(main, ['mtcnn', '-i', image_path, '-d', frames_path, '-o', output])
        assert result.exit_code != 0
        result = runner.invoke(main, ['mtcnn', '-o', output])
        assert result.exit_code != 0

def test_cli_mtcnn(image_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        output = os.path.join(temp_dir, 'out.json')
        result = runner.invoke(main, ['mtcnn', '-i', image_path, '--no-cuda', '-o', output])
        assert result.exit_code == 0

def test_cli_retinaface_image(image_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        arguments = [
            'retinaface', '-b', 'MOBILENET',
            '-i', image_path,
            '--no-cuda',
            '--print',
            '-o', os.path.join(temp_dir, 'out.json'),
            '-s', os.path.join(temp_dir, 'frames_out')
        ]
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0

def test_cli_retinaface_video(video_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        arguments = [
            'retinaface', '-b', 'MOBILENET',
            '-v', video_path,
            '--no-cuda',
            '-bs', '10',
            '-o', os.path.join(temp_dir, 'out.json'),
            '-s', os.path.join(temp_dir, 'frames_out')
        ]
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0

def test_cli_retinaface_frames(frames_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        runner = CliRunner()
        arguments = [
            'retinaface', '-b', 'MOBILENET',
            '-d', frames_path,
            '--no-cuda',
            '-bs', '3',
            '-o', os.path.join(temp_dir, 'out.json'),
        ]
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0

def test_cli_retinaface_list(frames_path):
    with tempfile.TemporaryDirectory() as temp_dir:

        with open(os.path.join(temp_dir, 'list.txt'), 'w') as pfile:
            for image_file in os.listdir(frames_path):
                pfile.write("%s\n" % os.path.join(frames_path, image_file))

        runner = CliRunner()
        arguments = [
            'retinaface', '-b', 'MOBILENET',
            '-l', os.path.join(temp_dir, 'list.txt'),
            '--no-cuda',
            '-bs', '3',
            '-o', os.path.join(temp_dir, 'out.json'),
        ]
        result = runner.invoke(main, arguments)
        assert result.exit_code == 0
