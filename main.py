import os
import copy
import onnx
import argparse
import subprocess

import torch
import torchaudio
import numpy as np
import onnxruntime as ort

import torch.utils.benchmark as benchmark
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor

from typing import Dict, Iterable

torch.manual_seed(0)

FRAME_SIZE = 480
INPUT_NAMES = [
    'input_frame',
    'states',
    'atten_lim_db'
]
OUTPUT_NAMES = [
    'enhanced_audio_frame', 'out_states', 'lsnr'
]


class TorchDFPipeline(nn.Module):
    def __init__(
            self, nb_bands=32, hop_size=480, fft_size=960,
            df_order=5, conv_lookahead=2, nb_df=96, model_base_dir='DeepFilterNet3',
            atten_lim_db=0.0, always_apply_all_stages=False, device='cpu'
    ):
        super().__init__()
        self.hop_size = hop_size
        self.fft_size = fft_size



        self.states = torch.zeros(45304, device=device)

        self.atten_lim_db = torch.tensor(atten_lim_db, device=device)

    def forward(self, input_audio: Tensor, sample_rate: int) -> Tensor:
        """
        Denoising audio frame using exportable fully torch model.

        Parameters:
            input_audio:      Float[1, t] - Input audio
            sample_rate:      Int - Sample rate

        Returns:
            enhanced_audio:   Float[1, t] - Enhanced input audio
        """
        assert input_audio.shape[0] == 1, f'Only mono supported! Got wrong shape! {input_audio.shape}'
        # assert sample_rate == self.sample_rate, f'Only {self.sample_rate} supported! Got wrong sample rate! {sample_rate}'

        input_audio = input_audio.squeeze(0)
        orig_len = input_audio.shape[0]
        
        # exit()
        hop_size_divisible_padding_size = (self.hop_size - orig_len % self.hop_size) % self.hop_size
       
        orig_len += hop_size_divisible_padding_size
        input_audio = F.pad(input_audio, (0, self.fft_size + hop_size_divisible_padding_size))

        chunked_audio = torch.split(input_audio, self.hop_size)

        

        output_frames = []

        for input_frame in chunked_audio:
            (
                enhanced_audio_frame, self.states, lsnr
            ) = self.torch_streaming_model(
                input_frame,
                self.states,
                self.atten_lim_db
            )
            
            output_frames.append(enhanced_audio_frame)

        enhanced_audio = torch.cat(output_frames).unsqueeze(0) # [t] -> [1, t] typical mono format

        d = self.fft_size - self.hop_size
        enhanced_audio = enhanced_audio[:, d : orig_len + d]

        return enhanced_audio

def generate_onnx_features(input_features):
    return {
        x: y.detach().cpu().numpy()
        for x, y in zip(INPUT_NAMES, input_features)
    }

def infer_onnx_model(streaming_pipeline, ort_session, inference_path):
    """
    Inference ONNX model with TorchDFPipeline
    """

    # del streaming_pipeline.torch_streaming_model
    streaming_pipeline.torch_streaming_model = lambda *features: (
        torch.from_numpy(x) for x in ort_session.run(
        OUTPUT_NAMES,
        generate_onnx_features(list(features)),
    )
    )

    noisy_audio, sr = torchaudio.load(inference_path, channels_first=True)
    # print("Audio sample rate: ", sr)  

    resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
    upsampled_waveform = resample(noisy_audio)

    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0) # stereo to mono

    enhanced_audio = streaming_pipeline(noisy_audio, sr)
    
    torchaudio.save(
        inference_path.replace('.wav', '_onnx_infer.wav'), enhanced_audio, sr,
        encoding="PCM_S", bits_per_sample=16
    )

def main():
    streaming_pipeline = TorchDFPipeline(always_apply_all_stages=True, device='cpu')

    model_path = "denoiser_model.onnx"
    print('Checking model...')
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = model_path
    sess_options.intra_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort_session = ort.InferenceSession(model_path, sess_options, providers=['CPUExecutionProvider'])

    inference_path = "noiserRemovalSample1.wav"
    infer_onnx_model(streaming_pipeline, ort_session, inference_path)


main()
