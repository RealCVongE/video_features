# 필요한 라이브러리와 모듈을 임포트합니다.
import os
import cv2
import numpy as np
import torch
import torchvision
from typing import Dict, List
from models.i3d.i3d_src.i3d_net import I3D  # I3D 모델 구현
from models._base.base_extractor import BaseExtractor  # 기본 특징 추출기 클래스를 상속
from models.raft.raft_src.raft import RAFT, InputPadder
from models.transforms import (Clamp, PILToTensor, ResizeImproved, ScaleTo1_1, TensorCenterCrop, ToFloat, PermuteAndUnsqueeze, ToUInt8)
from utils.utils import dp_state_to_normal, show_predictions_on_dataset
from models.raft.extract_raft import DATASET_to_RAFT_CKPT_PATHS
from video_feature.utils.io import reencode_video_with_diff_fps

# I3D 모델을 사용해 RGB 데이터만 추출하는 클래스
class ExtractI3D(BaseExtractor):
    def __init__(self, args) -> None:
        super().__init__(
            feature_type=args['feature_type'],
            on_extraction=args.get('on_extraction', 'save_numpy'),
            tmp_path=args['tmp_path'],
            output_path=args['output_path'],
            keep_tmp_files=args['keep_tmp_files'],
            device=args['device'],
        )
        self.flow_type = args.flow_type
        self.streams = ['rgb',"flow"]
        self.i3d_classes_num = 400
        self.min_side_size = 256
        self.central_crop_size = 224
        self.extraction_fps = args.get('extraction_fps', None)
        self.step_size = args.get('step_size', 16)
        self.stack_size = args.get('stack_size', 16)
        self.resize_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            ResizeImproved(self.min_side_size),
            PILToTensor(),
            ToFloat(),
        ])
        self.i3d_transforms = {
            'rgb': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_list: List) -> Dict[str, np.ndarray]:
        # if self.extraction_fps is not None:
            # video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)
        rgb_stack = []
        feats_dict = {'rgb': [],"flow":[]}
        padder = None
        first_frame = True
        for i in video_list :
            rgb = i

            # if first_frame:
            #     first_frame = False
            #     if not frame_exists:
            #         continue
            # if frame_exists:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = self.resize_transforms(rgb)
            rgb = rgb.unsqueeze(0)
            if self.flow_type == 'raft' and padder is None:
                padder = InputPadder(rgb.shape)

            rgb_stack.append(rgb)

            if len(rgb_stack) == self.stack_size + 1:
                batch_feats_dict = self.run_on_a_stack(rgb_stack,padder)
                for stream in self.streams:
                    feats_dict[stream].extend(batch_feats_dict[stream].tolist())
                rgb_stack = rgb_stack[self.step_size:]  # 스텝 사이즈만큼 스택에서 제거


        feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}
        return feats_dict
    # @torch.no_grad()
    # def extract(self, video_path: str) -> Dict[str, np.ndarray]:
    #     # if self.extraction_fps is not None:
    #         # video_path = reencode_video_with_diff_fps(video_path, self.tmp_path, self.extraction_fps)

    #     cap = cv2.VideoCapture(video_path)
    #     rgb_stack = []
    #     feats_dict = {'rgb': [],"flow":[]}
    #     padder = None
    #     first_frame = True
    #     while cap.isOpened():
    #         frame_exists, rgb = cap.read()

    #         if first_frame:
    #             first_frame = False
    #             if not frame_exists:
    #                 continue

    #         if frame_exists:
    #             rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    #             rgb = self.resize_transforms(rgb)
    #             rgb = rgb.unsqueeze(0)
    #             if self.flow_type == 'raft' and padder is None:
    #                 padder = InputPadder(rgb.shape)

    #             rgb_stack.append(rgb)

    #             if len(rgb_stack) == self.stack_size + 1:
    #                 batch_feats_dict = self.run_on_a_stack(rgb_stack,padder)
    #                 for stream in self.streams:
    #                     feats_dict[stream].extend(batch_feats_dict[stream].tolist())
    #                 rgb_stack = rgb_stack[self.step_size:]  # 스텝 사이즈만큼 스택에서 제거
    #         else:
    #             cap.release()
    #             break

    #     if (self.extraction_fps is not None) and (not self.keep_tmp_files):
    #         os.remove(video_path)

    #     feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}
    #     return feats_dict
    def run_on_a_stack(self, rgb_stack, padder=None) -> Dict[str, torch.Tensor]:
        models = self.name2module['model']
        flow_xtr_model = self.name2module.get('flow_xtr_model', None)
        rgb_stack = torch.cat(rgb_stack).to(self.device)

        batch_feats_dict = {}
        for stream in self.streams:
            # if i3d stream is flow, we first need to calculate optical flow, otherwise, we use rgb
            # `end_idx-1` and `start_idx+1` because flow is calculated between f and f+1 frames
            # we also use `end_idx-1` for stream == 'rgb' case: just to make sure the feature length
            # is same regardless of whether only rgb is used or flow
            if stream == 'flow':
                if self.flow_type == 'raft':
                    stream_slice = flow_xtr_model(padder.pad(rgb_stack)[:-1], padder.pad(rgb_stack)[1:])
                else:
                    raise NotImplementedError
            elif stream == 'rgb':
                stream_slice = rgb_stack[:-1]
            else:
                raise NotImplementedError
            # apply transforms depending on the stream (flow or rgb)
            stream_slice = self.i3d_transforms[stream](stream_slice)
            # extract features for a stream
            batch_feats_dict[stream] = models[stream](stream_slice, features=True)  # (B, 1024)
            # add features to the output dict

        return batch_feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.
        Since I3D is two-stream, it may load a optical flow extraction model as well.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        flow_model_paths = {'raft': os.path.join(os.path.dirname( os.path.abspath(__file__)),DATASET_to_RAFT_CKPT_PATHS['sintel']), }
        i3d_weights_paths = {
            'rgb': os.path.join(os.path.dirname( os.path.abspath(__file__)),'./models/i3d/checkpoints/i3d_rgb.pt'),
            'flow':os.path.join(os.path.dirname( os.path.abspath(__file__)), './models/i3d/checkpoints/i3d_flow.pt'),
        }
        name2module = {}

        if "flow" in self.streams:
            # Flow extraction module
            if self.flow_type == 'raft':
                flow_xtr_model = RAFT()
            else:
                raise NotImplementedError(f'Flow model {self.flow_type} is not implemented')
            # Preprocess state dict
            state_dict = torch.load(flow_model_paths[self.flow_type], map_location='cpu')
            state_dict = dp_state_to_normal(state_dict)
            flow_xtr_model.load_state_dict(state_dict)
            flow_xtr_model = flow_xtr_model.to(self.device)
            flow_xtr_model.eval()
            name2module['flow_xtr_model'] = flow_xtr_model

        # Feature extraction models (rgb and flow streams)
        i3d_stream_models = {}
        for stream in self.streams:
            i3d_stream_model = I3D(num_classes=self.i3d_classes_num, modality=stream)
            i3d_stream_model.load_state_dict(torch.load(i3d_weights_paths[stream], map_location='cpu'))
            i3d_stream_model = i3d_stream_model.to(self.device)
            i3d_stream_model.eval()
            i3d_stream_models[stream] = i3d_stream_model
        name2module['model'] = i3d_stream_models

        return name2module