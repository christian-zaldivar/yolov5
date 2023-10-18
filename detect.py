# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
from ultralytics.utils.plotting import save_one_box

from api.yolo.models.common import DetectMultiBackend
from api.yolo.utils.augmentations import letterbox
from api.yolo.utils.general import (
    Profile,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from api.yolo.utils.torch_utils import select_device, smart_inference_mode


WEIGHTS = Path("best.pt")


@lru_cache(10)
def _model(weights: Path, device: str, dnn: bool, fp16: bool):
    return DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=fp16)


@smart_inference_mode()
def run(
    image_: np.ndarray,
    add_padding: bool = False,
    weights: Path = WEIGHTS,  # model path or triton URL
    imgsz: tuple = (640, 640),  # inference size (height, width)
    conf_thres: float = 0.25,  # confidence threshold
    iou_thres: float = 0.45,  # NMS IOU threshold
    device: str = "",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    agnostic_nms: bool = False,  # class-agnostic NMS
    augment: bool = False,  # augmented inference
    half: bool = False,  # use FP16 half-precision inference
    dnn: bool = False,  # use OpenCV DNN for ONNX inference
) -> np.ndarray:
    """Detect signature and return the corresponding cropped image"""
    # Load model
    device = select_device(device)
    model = _model(weights, device, dnn, half)
    stride, pt = model.stride, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    dt = (Profile(), Profile(), Profile())
    im = letterbox(image_, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    # for path, im, im0s, _, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=augment, visualize=False)
    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, 1, agnostic_nms, max_det=1)[0]

    # Process prediction
    imc = image_.copy()  # for save_crop
    if len(pred):
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_boxes(im.shape[2:], pred[:, :4], image_.shape).round()

        # Write results
        *xyxy, _, _ = list(reversed(pred))[0]
        if add_padding:
            xyxy[0] = xyxy[0] - 40
            xyxy[1] = xyxy[1] - 40
            xyxy[2] = xyxy[2] + 40
            xyxy[3] = xyxy[3] + 40
        result = save_one_box(xyxy, imc, BGR=True, save=False)
        return result
