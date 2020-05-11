"""The MTCNN Detector"""

import math
import collections
from typing import Tuple, List, Optional#, Dict
import numpy as np
from PIL import Image
import torch
from torchvision.models.utils import load_state_dict_from_url
from fdet.detector import Detector, OutType


# pylint: disable=invalid-sequence-index
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments

class MTCNN(Detector):
    """MTCNN detector class based on work from:

    Zhang, K., Zhang, Z., Li, Z., and Qiao, Y. (2016).
    Joint face detection and alignment using multitask cascaded convolutional networks.
    IEEE Signal Processing Letters, 23(10):1499–1503.

    This implementations was based on:
    - https://github.com/TropComplique/mtcnn-pytorch/.
    - https://github.com/ipazc/mtcnn
    """

    def __init__(self, min_face_size: float = 20.0,
                 thresholds: Tuple[float, float, float] = (0.6, 0.7, 0.8),
                 nms_thresholds: Tuple[float, float, float] = (0.7, 0.7, 0.7),
                 cuda_benchmark: bool = True, cuda_devices: Optional[List[int]] = None,
                 cuda_enable: bool = torch.cuda.is_available()) -> None:
        """Initializes the MTCNN detector.

        :param min_face_size:  Minimum size of face to detect, in pixels. Defaults to 20.0.
        :type min_face_size: float, optional

        :param thresholds: The thresholds fo each MTCNN step. Defaults to (0.6, 0.7, 0.8).
        :type thresholds: Tuple[float, float, float], optional

        :param nms_thresholds: The NMS thresholds fo each MTCNN step. Defaults to (0.7, 0.7, 0.7).
        :type nms_thresholds: Tuple[float, float, float], optional

        :param cuda_benchmark: Indicates if the cuda_benchmark is enable or not. Defaults to True.
        :type cuda_benchmark: bool, optional

        :param cuda_devices: GPUs to be used. If None, uses all avaliable GPUs. Defaults to None.
        :type cuda_devices: Optional[List[int]], optional

        :param cuda_enable: Indicates if cuda should be used. Defaults to cuda.is_available().
        :type cuda_enable: bool, optional
        """
        Detector.__init__(self, cuda_devices, cuda_enable)

        self._min_face_size = min_face_size
        self._thresholds = thresholds
        self._nms_thresholds = nms_thresholds

        base_url = 'https://www.dropbox.com/s/'
        self._pnet = self.__load_model(_PNet, base_url + '1xi4gjoboaoa7e2/mtcnn_pnet.pt?dl=1')
        self._rnet = self.__load_model(_RNet, base_url + 'w6gqd6bxrjwh1ux/mtcnn_rnet.pt?dl=1')
        self._onet = self.__load_model(_ONet, base_url + 'hupifrhnigx89dp/mtcnn_onet.pt?dl=1')

        self._pnet = self._init_torch_module(self._pnet)
        self._rnet = self._init_torch_module(self._rnet)
        self._onet = self._init_torch_module(self._onet)

        torch.cuda.manual_seed(1137) # type: ignore
        torch.backends.cudnn.enabled = cuda_enable # type: ignore
        torch.backends.cudnn.benchmark = cuda_benchmark # type: ignore
        self._onet.eval()

    def _detect_batch(self, data: List[np.ndarray]) -> List[OutType]:

        frames = [Image.fromarray(frame) for frame in data]
        detections: List[OutType] = list()

        if not frames:
            return detections

        # ------------------------------------------------------------------------------------------
        # FIRST STAGE
        # ------------------------------------------------------------------------------------------
        all_frames_bboxes_candidates = self.__first_stage(
            frames, self._thresholds[0], self._nms_thresholds[0]
        )

        if all_frames_bboxes_candidates is None or not all_frames_bboxes_candidates:
            return detections

        # ------------------------------------------------------------------------------------------
        # SECOND STAGE
        # ------------------------------------------------------------------------------------------
        # Execute second_stage w/ candidate boxes of each frame
        all_frames_bboxes_candidates = self.__second_stage(
            frames, all_frames_bboxes_candidates, self._thresholds[1], self._nms_thresholds[1]
        )

        if self.__is_list_empty(all_frames_bboxes_candidates):
            return [[]]

        if all_frames_bboxes_candidates is None or not all_frames_bboxes_candidates:
            return detections

        # ------------------------------------------------------------------------------------------
        # THIRD STAGE
        # ------------------------------------------------------------------------------------------


        # Execute third_stage
        final_result = list()

        frames_bboxes_list, frames_landmarks_list = self.__third_stage(
            frames, all_frames_bboxes_candidates, self._thresholds[2], self._nms_thresholds[2])


        for frame_bboxes, frame_landmarks in zip(frames_bboxes_list, frames_landmarks_list):
            frame_detections: OutType = list()
            if frame_bboxes is not None:
                for bbox, keypoints in zip(frame_bboxes, frame_landmarks):
                    frame_detections.append({
                        'box': (int(bbox[0]), int(bbox[1]),
                                int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])),
                        'confidence': float(bbox[-1]),
                        'keypoints': {
                            'left_eye': (int(keypoints[0]), int(keypoints[5])),
                            'right_eye': (int(keypoints[1]), int(keypoints[6])),
                            'nose': (int(keypoints[2]), int(keypoints[7])),
                            'mouth_left': (int(keypoints[3]), int(keypoints[8])),
                            'mouth_right': (int(keypoints[4]), int(keypoints[9])),
                        }
                    })
            final_result.append(frame_detections)

        return final_result

    def __first_stage(self, frames: List, threshold: float,
                      nms_threshold: float) -> List[np.ndarray]:

        # BUILD AN IMAGE PYRAMID
        width, height = frames[0].size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales: List[float] = list()

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        min_length *= (min_detection_size/self._min_face_size)

        # prepare scales
        factor_count = 0
        while min_length > min_detection_size:
            scales.append((min_detection_size/self._min_face_size) * factor ** factor_count)
            min_length *= factor
            factor_count += 1

        # List to store the bboxes of pnet belonging to all detection frames
        all_frames_bboxes_list: List[List[np.ndarray]] = [list() for _ in range(0, len(frames))]

        # run PNet on different scales
        for scale in scales:

            # scale the image and convert it to a float array
            scaled_width, scaled_heigth = math.ceil(width * scale), math.ceil(height * scale)

            scaled_frame_list: List = list()
            for frame in frames:
                scaled_frame = frame.resize((scaled_width, scaled_heigth), Image.BILINEAR)
                scaled_frame = np.asarray(scaled_frame, 'float32')
                scaled_frame = _preprocess(scaled_frame, expand=False)
                scaled_frame_list.append(scaled_frame)

            np_scaled_frame_list = np.asarray(scaled_frame_list)
            torch_scaled_frame_list = torch.from_numpy(np_scaled_frame_list)
            torch_scaled_frame_list = torch_scaled_frame_list.to(self._device_control)

            output = self._pnet(torch_scaled_frame_list)

            # probs: probability of a face at each sliding window
            # offsets: transformations to true bounding boxes
            probs = output[1].cpu().data.numpy()[:, 1, :, :]
            offsets = output[0].cpu().data.numpy()[:, :, :, :]
            offsets = np.expand_dims(offsets, axis=1)

            # generate bboxes and offsets
            for idx, (prob, offset) in enumerate(zip(probs, offsets)):
                bboxes = _generate_bboxes(prob, offset, scale, threshold)

                if bboxes is None:
                    continue

                # nms on bboxes
                keep = _nms(bboxes[:, 0:5], overlap_threshold=0.5)
                all_frames_bboxes_list[idx].append(bboxes[keep])

        final_bboxes: List[np.ndarray] = list()

        for frame_bboxes in all_frames_bboxes_list:
            if frame_bboxes:
                frame_detections = np.vstack(frame_bboxes)
                keep_idx_boxes = _nms(frame_detections[:, 0:5], nms_threshold)
                frame_detections = frame_detections[keep_idx_boxes]
                frame_detections = _calibrate_box(frame_detections[:, 0:5], frame_detections[:, 5:])
                frame_detections = _convert_to_square(frame_detections)
                frame_detections[:, 0:4] = np.round(frame_detections[:, 0:4])
            else:
                frame_detections = None

            final_bboxes.append(frame_detections)

        return final_bboxes

    def __second_stage(self, frames: List, all_frames_bboxes_candidates: List[np.ndarray],
                       threshold: float, nms_threshold: float) -> np.ndarray:

        prev_idx = 0
        frames_bboxes_indexes_list: List[int] = list()
        frames_bboxes_indexes_list.insert(0, 0)
        for frame_bboxes in all_frames_bboxes_candidates:
            n_detections = frame_bboxes.shape[0] if frame_bboxes is not None else 0
            frames_bboxes_indexes_list.append(n_detections + prev_idx)
            prev_idx += n_detections

        frames_bboxes_indexes = list(
            zip(frames_bboxes_indexes_list[:-1], frames_bboxes_indexes_list[1:])
        )

        for idx, frame_bboxes in enumerate(all_frames_bboxes_candidates):
            if frame_bboxes is not None:
                all_frames_bboxes_candidates[idx][:, 0:4] = np.round(frame_bboxes[:, 0:4])

        all_frames_bboxes_ccat = [
            fbboxes for fbboxes in all_frames_bboxes_candidates
            if fbboxes is not None
        ]

        if not all_frames_bboxes_ccat:
            return [None] * len(all_frames_bboxes_candidates)

        all_frames_bboxes_ccat = np.concatenate(all_frames_bboxes_ccat, axis=0)

        all_frames_bboxes_images = [
            _get_image_boxes(all_frames_bboxes_ccat[first_idx:last_idx], frames[frame_idx], size=24)
            for frame_idx, (first_idx, last_idx) in enumerate(frames_bboxes_indexes)
        ]

        #  return all_frames_bboxes_candidates

        all_frames_bboxes_images_np = np.concatenate(all_frames_bboxes_images, axis=0)

        all_frames_bboxes_images_torch = torch.from_numpy(all_frames_bboxes_images_np)

        all_frames_bboxes_images_torch = all_frames_bboxes_images_torch.to(self._device_control)
        output = self._rnet(all_frames_bboxes_images_torch)

        output_offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
        output_probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        offsets_by_frame = [
            output_offsets[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        # return offsets_by_frame

        probs_by_frame = [
            output_probs[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        all_frames_bboxes_candidates = [
            all_frames_bboxes_ccat[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        final_frames_result = list()
        zip_iterable = zip(probs_by_frame, offsets_by_frame, all_frames_bboxes_candidates)
        for frame_probs, frame_offsets, frame_bboxes in zip_iterable:

            keep_bboxes: Optional[np.ndarray] = None

            if frame_bboxes is not None:
                keep = np.where(frame_probs[:, 1] > threshold)[0]
                keep_bboxes = frame_bboxes[keep]

                keep_bboxes[:, 4] = frame_probs[keep, 1].reshape((-1,))

                frame_offsets = frame_offsets[keep]

                keep = _nms(keep_bboxes, nms_threshold)

                keep_bboxes = keep_bboxes[keep]
                keep_bboxes = _calibrate_box(keep_bboxes, frame_offsets[keep])

                keep_bboxes = _convert_to_square(keep_bboxes)
                keep_bboxes[:, 0:4] = np.round(keep_bboxes[:, 0:4])
                #pylint: disable=unsubscriptable-object
                if keep_bboxes.shape[0] == 0:
                    keep_bboxes = None
                #pylint: enable=unsubscriptable-object

            final_frames_result.append(keep_bboxes)

        return final_frames_result

    def __third_stage(self, frames: List, all_frames_bboxes_candidates: List[np.ndarray],
                      threshold: float, nms_threshold: float) -> Tuple[np.ndarray, np.ndarray]:


        prev_idx = 0
        frames_bboxes_indexes_list: List[int] = list()
        frames_bboxes_indexes_list.insert(0, 0)
        for _, frame_bboxes in enumerate(all_frames_bboxes_candidates):
            n_detections = frame_bboxes.shape[0] if frame_bboxes is not None else 0
            frames_bboxes_indexes_list.append(n_detections + prev_idx)
            prev_idx += n_detections

        frames_bboxes_indexes = list(
            zip(frames_bboxes_indexes_list[:-1], frames_bboxes_indexes_list[1:])
        )

        all_frames_bboxes_ccat = [
            fbboxes for fbboxes in all_frames_bboxes_candidates
            if fbboxes is not None
        ]

        if not all_frames_bboxes_ccat:
            return [None] * len(frames_bboxes_indexes), [None] * len(frames_bboxes_indexes)

        all_frames_bboxes_ccat = np.concatenate(all_frames_bboxes_ccat, axis=0)

        all_frames_bboxes_images = [
            _get_image_boxes(all_frames_bboxes_ccat[first_idx:last_idx], frames[frame_idx], size=48)
            for frame_idx, (first_idx, last_idx) in enumerate(frames_bboxes_indexes)
        ]

        all_frames_bboxes_images_np = np.concatenate(all_frames_bboxes_images, axis=0)
        all_frames_bboxes_images_torch = torch.from_numpy(all_frames_bboxes_images_np)
        all_frames_bboxes_images_torch = all_frames_bboxes_images_torch.to(self._device_control)

        output = self._onet(all_frames_bboxes_images_torch)

        output_landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        output_offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
        output_probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        offsets_by_frame = [
            output_offsets[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        probs_by_frame = [
            output_probs[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        landmarks_by_frame = [
            output_landmarks[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        all_frames_bboxes_candidates = [
            all_frames_bboxes_ccat[first_idx:last_idx] if first_idx != last_idx else None
            for first_idx, last_idx in frames_bboxes_indexes
        ]

        # Iterate each result per frame and compute NMS and remove low-score bboxes
        temp_zip = zip(landmarks_by_frame, probs_by_frame, offsets_by_frame,
                       all_frames_bboxes_candidates)
        final_frames_result = list()
        final_landmarks_list = list()
        for frame_landmarks, frame_probs, frame_offsets, frame_bboxes in temp_zip:

            keep_bboxes = None

            if frame_bboxes is not None:
                keep = np.where(frame_probs[:, 1] > threshold)[0]
                keep_bboxes = frame_bboxes[keep]
                keep_bboxes[:, 4] = frame_probs[keep, 1].reshape((-1,))
                frame_offsets = frame_offsets[keep]
                frame_landmarks = frame_landmarks[keep]

                width = keep_bboxes[:, 2] - keep_bboxes[:, 0] + 1.0
                height = keep_bboxes[:, 3] - keep_bboxes[:, 1] + 1.0
                xmin, ymin = keep_bboxes[:, 0], keep_bboxes[:, 1]
                frame_landmarks[:, 0:5] = np.expand_dims(
                    xmin, 1) + np.expand_dims(width, 1)*frame_landmarks[:, 0:5]
                frame_landmarks[:, 5:10] = np.expand_dims(
                    ymin, 1) + np.expand_dims(height, 1)*frame_landmarks[:, 5:10]

                keep_bboxes = _calibrate_box(keep_bboxes, frame_offsets)
                keep = _nms(keep_bboxes, nms_threshold, mode='min')
                keep_bboxes = keep_bboxes[keep]
                frame_landmarks = frame_landmarks[keep]

                if keep_bboxes.shape[0] == 0:
                    keep_bboxes = []

            final_frames_result.append(keep_bboxes)
            final_landmarks_list.append(frame_landmarks)

        return final_frames_result, final_landmarks_list

    def __is_list_empty(self, in_list):
        if isinstance(in_list, list):  # Is a list
            return all(map(self.__is_list_empty, in_list))
        return False  # Not a list

    def __load_model(self, net_class: type, url: str) -> torch.nn.Module:
        """Download and construct the models"""
        state_dict = load_state_dict_from_url(url, map_location=self._device_control)
        model = net_class()
        model.load_state_dict(state_dict, strict=False)
        return model


####################################################################################################
# MTCNN models - Extracted from: https://github.com/TropComplique/mtcnn-pytorch/
####################################################################################################

class _Flatten(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

    #pylint: disable=arguments-differ
    def forward(self, x):
        # without this pretrained model isn't working
        x = x.transpose(3, 2).contiguous()
        return x.view(x.size(0), -1)
    #pylint: enable=arguments-differ

class _PNet(torch.nn.Module):

    def __init__(self):

        torch.nn.Module.__init__(self)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 10, 3, 1)),
            ('prelu1', torch.nn.PReLU(10)),
            ('pool1', torch.nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv2', torch.nn.Conv2d(10, 16, 3, 1)),
            ('prelu2', torch.nn.PReLU(16)),

            ('conv3', torch.nn.Conv2d(16, 32, 3, 1)),
            ('prelu3', torch.nn.PReLU(32))
        ]))

        self.conv4_1 = torch.nn.Conv2d(32, 2, 1, 1)
        self.conv4_2 = torch.nn.Conv2d(32, 4, 1, 1)

    #pylint: disable=arguments-differ
    #pylint: disable=invalid-name
    def forward(self, x):
        x = self.features(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = torch.nn.functional.softmax(a, dim=1)
        return b, a
    #pylint: enable=invalid-name
    #pylint: enable=arguments-differ

class _RNet(torch.nn.Module):
    """RNet"""

    def __init__(self):

        torch.nn.Module.__init__(self)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 28, 3, 1)),
            ('prelu1', torch.nn.PReLU(28)),
            ('pool1', torch.nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', torch.nn.Conv2d(28, 48, 3, 1)),
            ('prelu2', torch.nn.PReLU(48)),
            ('pool2', torch.nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', torch.nn.Conv2d(48, 64, 2, 1)),
            ('prelu3', torch.nn.PReLU(64)),

            ('flatten', _Flatten()),
            ('conv4', torch.nn.Linear(576, 128)),
            ('prelu4', torch.nn.PReLU(128))
        ]))

        self.conv5_1 = torch.nn.Linear(128, 2)
        self.conv5_2 = torch.nn.Linear(128, 4)

        # this_file_path = os.path.dirname(os.path.abspath(__file__))
        # pnet_weights_path = os.path.join(this_file_path, 'weights', 'rnet.npy')
        # weights = np.load(pnet_weights_path, allow_pickle=True)[()]
        # for n, p in self.named_parameters():
        #     p.data = torch.FloatTensor(weights[n])

    #pylint: disable=arguments-differ
    #pylint: disable=invalid-name
    def forward(self, x):
        x = self.features(x)
        a = self.conv5_1(x)
        b = self.conv5_2(x)
        a = torch.nn.functional.softmax(a, dim=1)
        return b, a
    #pylint: enable=invalid-name
    #pylint: enable=arguments-differ

class _ONet(torch.nn.Module):

    def __init__(self):

        torch.nn.Module.__init__(self)

        self.features = torch.nn.Sequential(collections.OrderedDict([
            ('conv1', torch.nn.Conv2d(3, 32, 3, 1)),
            ('prelu1', torch.nn.PReLU(32)),
            ('pool1', torch.nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv2', torch.nn.Conv2d(32, 64, 3, 1)),
            ('prelu2', torch.nn.PReLU(64)),
            ('pool2', torch.nn.MaxPool2d(3, 2, ceil_mode=True)),

            ('conv3', torch.nn.Conv2d(64, 64, 3, 1)),
            ('prelu3', torch.nn.PReLU(64)),
            ('pool3', torch.nn.MaxPool2d(2, 2, ceil_mode=True)),

            ('conv4', torch.nn.Conv2d(64, 128, 2, 1)),
            ('prelu4', torch.nn.PReLU(128)),

            ('flatten', _Flatten()),
            ('conv5', torch.nn.Linear(1152, 256)),
            ('drop5', torch.nn.Dropout(0.25)),
            ('prelu5', torch.nn.PReLU(256)),
        ]))

        self.conv6_1 = torch.nn.Linear(256, 2)
        self.conv6_2 = torch.nn.Linear(256, 4)
        self.conv6_3 = torch.nn.Linear(256, 10)

    #pylint: disable=arguments-differ
    #pylint: disable=invalid-name
    def forward(self, x):
        x = self.features(x)
        a = self.conv6_1(x)
        b = self.conv6_2(x)
        c = self.conv6_3(x)
        a = torch.nn.functional.softmax(a, dim=1)
        return c, b, a
    #pylint: enable=invalid-name
    #pylint: enable=arguments-differ


####################################################################################################
# BOX Utils - Based on: https://github.com/TropComplique/mtcnn-pytorch/
####################################################################################################
#pylint: disable=invalid-name
def _preprocess(img, expand=True):
    if expand:
        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5)*0.0078125
    else:
        img = img.transpose((2, 0, 1))
        img = (img - 127.5)*0.0078125
    return img

def _generate_bboxes(probs: np.ndarray, offsets: np.ndarray,
                     scale: float, threshold: float) -> Optional[np.ndarray]:
    """Generate bounding boxes at places
    where there is probably a face.
    Arguments:
        probs: a float numpy array of shape [n, m].
        offsets: a float numpy array of shape [1, 4, n, m].
        scale: a float number,
            width and height of the image were scaled by this number.
        threshold: a float number.
    Returns:
        a float numpy array of shape [n_boxes, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = np.where(probs > threshold)

    if inds[0].size == 0:
        return None

    # transformations of bounding boxes
    tx1, ty1, tx2, ty2 = [offsets[0, i, inds[0], inds[1]] for i in range(4)]
    # they are defined as:
    # w = x2 - x1 + 1
    # h = y2 - y1 + 1
    # x1_true = x1 + tx1*w
    # x2_true = x2 + tx2*w
    # y1_true = y1 + ty1*h
    # y2_true = y2 + ty2*h

    offsets = np.array([tx1, ty1, tx2, ty2])
    score = probs[inds[0], inds[1]]

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    bounding_boxes = np.vstack([
        np.round((stride*inds[1] + 1.0)/scale),
        np.round((stride*inds[0] + 1.0)/scale),
        np.round((stride*inds[1] + 1.0 + cell_size)/scale),
        np.round((stride*inds[0] + 1.0 + cell_size)/scale),
        score, offsets
    ])
    # why one is added?

    return bounding_boxes.T

def _nms(boxes, overlap_threshold=0.5, mode='union'):
    """Non-maximum suppression.

    Arguments:
        boxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        overlap_threshold: a float number.
        mode: 'union' or 'min'.

    Returns:
        list with indices of the selected boxes
    """

    # if there are no boxes, return the empty list
    if len(boxes) == 0:
        return []

    # list of picked indices
    pick = []

    # grab the coordinates of the bounding boxes
    x1, y1, x2, y2, score = [boxes[:, i] for i in range(5)]

    area = (x2 - x1 + 1.0)*(y2 - y1 + 1.0)
    ids = np.argsort(score)  # in increasing order

    while len(ids) > 0:

        # grab index of the largest value
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # compute intersections
        # of the box with the largest score
        # with the rest of boxes

        # left top corner of intersection boxes
        ix1 = np.maximum(x1[i], x1[ids[:last]])
        iy1 = np.maximum(y1[i], y1[ids[:last]])

        # right bottom corner of intersection boxes
        ix2 = np.minimum(x2[i], x2[ids[:last]])
        iy2 = np.minimum(y2[i], y2[ids[:last]])

        # width and height of intersection boxes
        w = np.maximum(0.0, ix2 - ix1 + 1.0)
        h = np.maximum(0.0, iy2 - iy1 + 1.0)

        # intersections' areas
        inter = w * h
        if mode == 'min':
            overlap = inter/np.minimum(area[i], area[ids[:last]])
        elif mode == 'union':
            # intersection over union (IoU)
            overlap = inter/(area[i] + area[ids[:last]] - inter)

        # delete all boxes where overlap is too big
        ids = np.delete(
            ids,
            np.concatenate([[last], np.where(overlap > overlap_threshold)[0]])
        )

    return pick

def _calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 5].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = np.expand_dims(w, 1)
    h = np.expand_dims(h, 1)

    # this is what happening here:
    # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
    # x1_true = x1 + tx1*w
    # y1_true = y1 + ty1*h
    # x2_true = x2 + tx2*w
    # y2_true = y2 + ty2*h
    # below is just more compact form of this

    # are offsets always such that
    # x1 < x2 and y1 < y2 ?

    translation = np.hstack([w, h, w, h])*offsets
    bboxes[:, 0:4] = bboxes[:, 0:4] + translation
    return bboxes

def _convert_to_square(bboxes) -> np.ndarray:
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 5].

    Returns:
        a float numpy array of shape [n, 5],
            squared bounding boxes.
    """
    square_bboxes = np.zeros_like(bboxes)
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = np.maximum(h, w)
    square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
    square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
    square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
    square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0
    return square_bboxes

def _get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.

    Arguments:
        bounding_boxes: a float numpy array of shape [n, 5].
        img: an instance of PIL.Image.
        size: an integer, size of cutouts.

    Returns:
        a float numpy array of shape [n, 3, size, size].
    """
    num_boxes = len(bounding_boxes)
    width, height = img.size

    [dy, edy, dx, edx, y, ey, x, ex, w, h] = _correct_bboxes(bounding_boxes, width, height)
    img_boxes = np.zeros((num_boxes, 3, size, size), 'float32')

    for i in range(num_boxes):
        img_box = np.zeros((h[i], w[i], 3), 'uint8')
        img_array = np.asarray(img, 'uint8')
        img_box[dy[i]:(edy[i] + 1), dx[i]:(edx[i] + 1), :] =\
            img_array[y[i]:(ey[i] + 1), x[i]:(ex[i] + 1), :]

        # resize
        img_box = Image.fromarray(img_box)
        img_box = img_box.resize((size, size), Image.BILINEAR)
        img_box = np.asarray(img_box, 'float32')

        img_boxes[i, :, :, :] = _preprocess(img_box)

    return img_boxes

def _correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        dy, dx, edy, edx: a int numpy arrays of shape [n],
            coordinates of the boxes with respect to the cutouts.
        y, x, ey, ex: a int numpy arrays of shape [n],
            corrected ymin, xmin, ymax, xmax.
        h, w: a int numpy arrays of shape [n],
            just heights and widths of boxes.

        in the following order:
            [dy, edy, dx, edx, y, ey, x, ex, w, h].
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w, h = x2 - x1 + 1.0, y2 - y1 + 1.0
    num_boxes = bboxes.shape[0]

    # 'e' stands for end
    # (x, y) -> (ex, ey)
    x, y, ex, ey = x1, y1, x2, y2

    # we need to cut out a box from the image.
    # (x, y, ex, ey) are corrected coordinates of the box
    # in the image.
    # (dx, dy, edx, edy) are coordinates of the box in the cutout
    # from the image.
    dx, dy = np.zeros((num_boxes,)), np.zeros((num_boxes,))
    edx, edy = w.copy() - 1.0, h.copy() - 1.0

    # if box's bottom right corner is too far right
    ind = np.where(ex > width - 1.0)[0]
    edx[ind] = w[ind] + width - 2.0 - ex[ind]
    ex[ind] = width - 1.0

    # if box's bottom right corner is too low
    ind = np.where(ey > height - 1.0)[0]
    edy[ind] = h[ind] + height - 2.0 - ey[ind]
    ey[ind] = height - 1.0

    # if box's top left corner is too far left
    ind = np.where(x < 0.0)[0]
    dx[ind] = 0.0 - x[ind]
    x[ind] = 0.0

    # if box's top left corner is too high
    ind = np.where(y < 0.0)[0]
    dy[ind] = 0.0 - y[ind]
    y[ind] = 0.0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, w, h]
    return_list = [i.astype('int32') for i in return_list]

    return return_list
