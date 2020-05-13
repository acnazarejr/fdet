"""The RetinaFace Detector"""

from typing import Optional, List, Sequence
from itertools import product
import math
import numpy as np
import torch
import cv2
import torchvision.models._utils
from torchvision.ops.boxes import batched_nms
import torchvision.models
from torchvision.models.utils import load_state_dict_from_url
from fdet.detector import Detector, SingleDetType
from fdet.utils.errors import DetectorValueError

#pylint: disable=too-many-arguments
#pylint: disable=too-many-locals

class RetinaFace(Detector):
    """RetinaFace detector class based on work from:

    Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou (2019).
    RetinaFace: Single-stage Dense Face Localisation in the Wild.
    arXiv, https://arxiv.org/abs/1905.00641

    This implementations was based on:
    - https://github.com/biubug6/Pytorch_Retinaface
    """

    @staticmethod
    def valid_backbones() -> List[str]:
        """List of valid backbones

        Returns:
            List[str]: A list with currently valid backbones.
        """
        return ['RESNET50', 'MOBILENET']

    def __init__(self, backbone: str, threshold: float = 0.8,
                 nms_threshold: float = 0.4, max_face_size: int = 1000,
                 cuda_enable: bool = torch.cuda.is_available(),
                 cuda_devices: Optional[Sequence[int]] = None, cuda_benchmark: bool = True) -> None:
        """Initializes the MTCNN detector.

        Args:
            backbone (str, optional): The backbone model [RESNET50 or MOBILENET]. Defaults to
                'RESNET50'.
            threshold (float, optional): The detection threshold. Defaults to 0.8.
            nms_threshold (float, optional): The nms threshold. Defaults to 0.4.
            max_face_size (int, optional): [description]. Defaults to 1000.
            cuda_enable (bool, optional): Indicates if cuda should be used. Defaults to
                cuda.is_available().
            cuda_devices (Optional[List[int]], optional): CUDA GPUs to be used. If None, uses all
                avaliable GPUs. Defaults to None.
            cuda_benchmark (bool, optional): [description]. Indicates if the cuda_benchmark is
                enable or not. Defaults to True.
        """

        Detector.__init__(self, cuda_enable=cuda_enable, cuda_devices=cuda_devices,
                          cuda_benchmark=cuda_benchmark)

        if (not isinstance(backbone, str)) or (backbone not in self.valid_backbones()):
            raise DetectorValueError('Invalid backbone: ' + str(backbone))
        self._net = self._init_torch_module(self.__load_retina_module(backbone=backbone))

        if threshold < 0.0 or threshold > 1.0:
            raise DetectorValueError('The threshold value must be between 0 and 1.')
        self._threshold = threshold

        if nms_threshold < 0.0 or nms_threshold > 1.0:
            raise DetectorValueError('The nms_threshold value must be between 0 and 1.')
        self._nms_threshold = nms_threshold

        if max_face_size < 100 or max_face_size > 1000:
            raise DetectorValueError('The max_face_size argument must be between 100 and 1000.')
        self._max_face_size = max_face_size

        torch.cuda.manual_seed(1137) # type: ignore
        torch.backends.cudnn.enabled = cuda_enable # type: ignore
        torch.backends.cudnn.benchmark = cuda_benchmark # type: ignore
        self._net.eval()

    def _run_data_batch(self, data: np.ndarray) -> List[List[SingleDetType]]:

        _, im_height, im_width, _ = data.shape
        max_axis = max(im_height, im_width)
        rescale = 1
        if max_axis > self._max_face_size:
            rescale = self._max_face_size/max_axis
            data = np.asarray([_image_rescale(image, rescale) for image in data])

        _, im_height, im_width, _ = data.shape
        images_np = np.asarray([np.float32(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)) for img in data])
        images_np -= (104, 117, 123)
        images_np = images_np.transpose(0, 3, 1, 2)
        images_torch = torch.from_numpy(images_np)
        images_torch = images_torch.to(self._device_control)

        scale_loc = torch.Tensor([im_width, im_height]).repeat(2) #type: ignore
        scale_loc = scale_loc.to(self._device_control)
        scale_landm = torch.Tensor([im_width, im_height]).repeat(5) #type: ignore
        scale_landm = scale_landm.to(self._device_control)

        priorbox = _PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self._device_control)
        prior_data = priors.data

        batch_loc, batch_conf, batch_landms = self._net(images_torch)  # forward pass

        n_images, n_boxes, _ = batch_loc.shape
        batch_idx = torch.arange(n_images).repeat_interleave(n_boxes).to(self._device_control)

        batch_scores = batch_conf[:, :, 1]
        batch_scores = batch_scores.reshape((-1))

        batch_boxes = _decode_loc(batch_loc.data, prior_data, variances=[0.1, 0.2])
        batch_boxes = batch_boxes * (scale_loc / rescale)
        batch_boxes = batch_boxes.reshape(n_images * n_boxes, 4)

        batch_landms = _decode_landm(batch_landms.data, prior_data, variances=[0.1, 0.2])
        batch_landms = batch_landms * (scale_landm / rescale)
        batch_landms = batch_landms.reshape(n_images * n_boxes, 10)

        mask_threshold = batch_scores.ge(self._threshold)
        batch_boxes = batch_boxes[mask_threshold]
        batch_idx = batch_idx[mask_threshold]
        batch_landms = batch_landms[mask_threshold]
        batch_scores = batch_scores[mask_threshold]

        keep = batched_nms(batch_boxes, batch_scores, batch_idx, self._nms_threshold)
        batch_boxes = batch_boxes[keep]
        batch_idx = batch_idx[keep]
        batch_landms = batch_landms[keep]
        batch_scores = batch_scores[keep]


        batch_detections = list()
        for idx in range(n_images):

            mask_current_image = batch_idx.eq(idx)

            boxes = batch_boxes[mask_current_image].data.cpu().numpy()
            scores = batch_scores[mask_current_image].data.cpu().numpy()
            landms = batch_landms[mask_current_image].data.cpu().numpy()

            dets = np.hstack((boxes, scores[:, np.newaxis], landms))
            batch_detections.append([_make_dict(det) for det in dets])


        return batch_detections


    def __load_retina_module(self, backbone: str) -> '_RetinaModule':

        url = ''
        if backbone == 'MOBILENET':
            url = 'https://www.dropbox.com/s/kr1xjmzry4l8p6g/retinaface_mobilenetv1_final.pt?dl=1'
        else: #if backbone == 'RESNET50':
            url = 'https://www.dropbox.com/s/d0xdha71fwr53uk/retinaface_resnet50_final.pt?dl=1'

        state_dict = load_state_dict_from_url(url, map_location=self._device_control)

        model = _RetinaModule(device_control=self._device_control, backbone=backbone)
        model.load_state_dict(state_dict, strict=False)
        return model

def _decode_loc(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors, 4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), dim=2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes

def _decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, :, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, :, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, :, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, :, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=2)
    return landms

def _make_dict(det) -> SingleDetType:
    return {
        'box': [int(det[0]), int(det[1]), int(det[2] - det[0]), int(det[3] - det[1])],
        'confidence': float(det[4]),
        'keypoints': {
            'left_eye': [int(det[5]), int(det[6])],
            'right_eye': [int(det[7]), int(det[8])],
            'nose': [int(det[9]), int(det[10])],
            'mouth_left': [int(det[11]), int(det[12])],
            'mouth_right': [int(det[13]), int(det[14])],
        }
    }

def _image_rescale(image: np.ndarray, scale: float) -> np.ndarray:
    img_height, img_width = image.shape[:2]
    size = int(img_width*scale), int(img_height*scale)
    return cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)

####################################################################################################
# RetinaFace models - Extracted from: https://github.com/biubug6/Pytorch_Retinaface
####################################################################################################

#pylint: disable=invalid-name
#pylint: disable=arguments-differ
#pylint: disable=too-many-instance-attributes

def _conv_bn(inp, oup, stride=1, leaky=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
        torch.nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def _conv_bn_no_relu(inp, oup, stride):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        torch.nn.BatchNorm2d(oup),
    )

def _conv_bn1X1(inp, oup, stride, leaky=0):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        torch.nn.BatchNorm2d(oup),
        torch.nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def _conv_dw(inp, oup, stride, leaky=0.1):
    return torch.nn.Sequential(
        torch.nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        torch.nn.BatchNorm2d(inp),
        torch.nn.LeakyReLU(negative_slope=leaky, inplace=True),

        torch.nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(oup),
        torch.nn.LeakyReLU(negative_slope=leaky, inplace=True),
    )

class _SSH(torch.nn.Module):

    def __init__(self, in_channel, out_channel):
        torch.nn.Module.__init__(self)
        assert out_channel % 4 == 0
        leaky = 0
        if out_channel <= 64:
            leaky = 0.1
        self.conv3X3 = _conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = _conv_bn(in_channel, out_channel//4, stride=1, leaky=leaky)
        self.conv5X5_2 = _conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = _conv_bn(out_channel//4, out_channel//4, stride=1, leaky=leaky)
        self.conv7x7_3 = _conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, _input):
        """forward"""
        conv3X3 = self.conv3X3(_input)

        conv5X5_1 = self.conv5X5_1(_input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = torch.nn.functional.relu(out)
        return out

class _FPN(torch.nn.Module):
    """FPN"""

    def __init__(self, in_channels_list, out_channels):
        torch.nn.Module.__init__(self)
        leaky = 0
        if out_channels <= 64:
            leaky = 0.1
        self.output1 = _conv_bn1X1(in_channels_list[0], out_channels, stride=1, leaky=leaky)
        self.output2 = _conv_bn1X1(in_channels_list[1], out_channels, stride=1, leaky=leaky)
        self.output3 = _conv_bn1X1(in_channels_list[2], out_channels, stride=1, leaky=leaky)

        self.merge1 = _conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = _conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, _input):
        """forward"""
        # names = list(_input.keys())
        _input = list(_input.values())

        output1 = self.output1(_input[0])
        output2 = self.output2(_input[1])
        output3 = self.output3(_input[2])

        up3 = torch.nn.functional.interpolate(output3, size=[output2.size(2), output2.size(3)],
                                              mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = torch.nn.functional.interpolate(output2, size=[output1.size(2), output1.size(3)],
                                              mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out

class _MobileNetV1(torch.nn.Module):
    """MobileNetV1"""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.stage1 = torch.nn.Sequential(
            _conv_bn(3, 8, 2, leaky=0.1),    # 3
            _conv_dw(8, 16, 1),   # 7
            _conv_dw(16, 32, 2),  # 11
            _conv_dw(32, 32, 1),  # 19
            _conv_dw(32, 64, 2),  # 27
            _conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = torch.nn.Sequential(
            _conv_dw(64, 128, 2),  # 43 + 16 = 59
            _conv_dw(128, 128, 1), # 59 + 32 = 91
            _conv_dw(128, 128, 1), # 91 + 32 = 123
            _conv_dw(128, 128, 1), # 123 + 32 = 155
            _conv_dw(128, 128, 1), # 155 + 32 = 187
            _conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = torch.nn.Sequential(
            _conv_dw(128, 256, 2), # 219 +3 2 = 241
            _conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(256, 1000)

    def forward(self, x):
        """forward"""
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

class _ClassHead(torch.nn.Module):
    """ClassHead"""

    def __init__(self, inchannels=512, num_anchors=3):
        torch.nn.Module.__init__(self)
        self.num_anchors = num_anchors
        self.conv1x1 = torch.nn.Conv2d(inchannels, self.num_anchors*2, kernel_size=(1, 1),
                                       stride=1, padding=0)

    def forward(self, x):
        """forward"""
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 2)

class _BboxHead(torch.nn.Module):
    """BboxHead"""

    def __init__(self, inchannels=512, num_anchors=3):
        torch.nn.Module.__init__(self)
        self.conv1x1 = torch.nn.Conv2d(inchannels, num_anchors*4, kernel_size=(1, 1), stride=1,
                                       padding=0)

    def forward(self, x):
        """forward"""
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()
        return out.view(out.shape[0], -1, 4)

class _LandmarkHead(torch.nn.Module):
    """LandmarkHead"""

    def __init__(self, inchannels=512, num_anchors=3):
        torch.nn.Module.__init__(self)
        self.conv1x1 = torch.nn.Conv2d(inchannels, num_anchors*10, kernel_size=(1, 1),
                                       stride=1, padding=0)

    def forward(self, x):
        """forward"""
        out = self.conv1x1(x)
        out = out.permute(0, 2, 3, 1).contiguous()

        return out.view(out.shape[0], -1, 10)

class _RetinaModule(torch.nn.Module):

    __configs = {

        'MOBILENET': {
            'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
            'in_channel': 32,
            'out_channel': 64
        },

        'RESNET50': {
            'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
            'in_channel': 256,
            'out_channel': 256
        },

    }

    def __init__(self, device_control: torch.device, backbone: str = 'RESNET50'):
        """
        :param cfg:  Network related settings.
        """
        torch.nn.Module.__init__(self)

        self._device_control = device_control

        config = _RetinaModule.__configs[backbone]
        backbone_model = None
        if backbone == 'MOBILENET':
            backbone_model = self._load_mobile_net_model()
        elif backbone == 'RESNET50':
            backbone_model = torchvision.models.resnet50(pretrained=True)

        #pylint: disable=protected-access
        self.body = torchvision.models._utils.IntermediateLayerGetter(
            backbone_model, config['return_layers'])
        #pylint: enable=protected-access
        in_channels_list = [
            config['in_channel'] * 2, #type: ignore
            config['in_channel'] * 4, #type: ignore
            config['in_channel'] * 8, #type: ignore
        ]
        out_channels = config['out_channel']
        self.fpn = _FPN(in_channels_list, out_channels)
        self.ssh1 = _SSH(out_channels, out_channels)
        self.ssh2 = _SSH(out_channels, out_channels)
        self.ssh3 = _SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=config['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=config['out_channel'])
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=config['out_channel'])

    def _load_mobile_net_model(self):
        model = _MobileNetV1()
        state_dict = load_state_dict_from_url(
            'https://www.dropbox.com/s/bd1keyo085pscfu/mobilenetv1_pretrain.pt?dl=1',
            map_location=self._device_control
        )
        # # load params
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _make_class_head(fpn_num=3, inchannels=64, anchor_num=2):
        classhead = torch.nn.ModuleList()
        for _ in range(fpn_num):
            classhead.append(_ClassHead(inchannels, anchor_num))
        return classhead

    @staticmethod
    def _make_bbox_head(fpn_num=3, inchannels=64, anchor_num=2):
        bboxhead = torch.nn.ModuleList()
        for _ in range(fpn_num):
            bboxhead.append(_BboxHead(inchannels, anchor_num))
        return bboxhead

    @staticmethod
    def _make_landmark_head(fpn_num=3, inchannels=64, anchor_num=2):
        landmarkhead = torch.nn.ModuleList()
        for _ in range(fpn_num):
            landmarkhead.append(_LandmarkHead(inchannels, anchor_num))
        return landmarkhead

    def forward(self, inputs):
        """forward"""
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat(
            [self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat(
            [self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat(
            [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        return (bbox_regressions, torch.nn.functional.softmax(classifications, dim=-1),
                ldm_regressions)

#pylint: disable=too-few-public-methods
class _PriorBox():
    """PrioBox class"""

    def __init__(self, image_size=None):
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [
            [math.ceil(self.image_size[0]/step), math.ceil(self.image_size[1]/step)]
            for step in self.steps
        ]
        self.name = "s"

    def forward(self):
        """forward"""
        anchors = []
        for k, feature_map in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for c_y, c_x in product(dense_cy, dense_cx):
                        anchors += [c_x, c_y, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
