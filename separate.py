#!/usr/bin/env python
import os
import sys
import warnings

import torch
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
from utils import spec_utils

warnings.filterwarnings("ignore")
cpu = torch.device('cpu')

import datetime

def print_with_timestamp(message):
    timestamp = datetime.datetime.now().isoformat()
    print(f"{timestamp} - {message}")

class Separator:
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file))[0]

        self.model_path = 'models/UVR_MDXNET_KARA_2.onnx'

        self.wav_type_set = "PCM_16"
        self.is_normalization = False
        self.is_denoise = False

        # These params are specific to the UVR_MDXNET_KARA_2 model
        # Fetched from model_data.json in UVR/models for model with hash 1d64a6d2c30f709b8c9b4ce1366d96ee
        self.compensate = 1.035
        self.dim_f, self.dim_t = 2048, 2**8
        self.n_fft = 5120

        self.chunks = 0
        self.margin = 44100
        self.adjust = 1
        self.dim_c = 4
        self.hop = 1024

        self.primary_source = None
        self.secondary_source = None

        self.device, self.run_type = torch.device('cpu'), ['CPUExecutionProvider']

    def write_audio(self, stem_path, stem_source, samplerate):
        sf.write(stem_path, stem_source, samplerate, subtype=self.wav_type_set)

    def separate(self):
        print_with_timestamp('Loading model...')
        samplerate = 44100

        ort_ = ort.InferenceSession(self.model_path, providers=self.run_type)
        self.model_run = lambda spek:ort_.run(None, {'input': spek.cpu().numpy()})[0]

        self.initialize_model_settings()
        print_with_timestamp('Running inference...')
        mdx_net_cut = True
        mix, raw_mix, samplerate = prepare_mix(self.audio_file, self.chunks, self.margin, mdx_net_cut=mdx_net_cut)
        source = self.demix_base(mix)[0]

        print_with_timestamp(f'Saving Instrumental stem...')
        primary_stem_path = os.path.join(f'{self.audio_file_base}_(Instrumental).wav')
        if not isinstance(self.primary_source, np.ndarray):
            self.primary_source = spec_utils.normalize(source, self.is_normalization).T
        self.write_audio(primary_stem_path, self.primary_source, samplerate)

        print_with_timestamp(f'Saving Vocals stem...')
        secondary_stem_path = os.path.join(f'{self.audio_file_base}_(Vocals).wav')
        if not isinstance(self.secondary_source, np.ndarray):
            raw_mix = self.demix_base(raw_mix, is_match_mix=True)[0] if mdx_net_cut else raw_mix
            self.secondary_source, raw_mix = spec_utils.normalize_two_stem(source*self.compensate, raw_mix, self.is_normalization)
            self.secondary_source = (-self.secondary_source.T+raw_mix.T)
        self.write_audio(secondary_stem_path, self.secondary_source, samplerate)

        torch.cuda.empty_cache()

    def initialize_model_settings(self):
        self.n_bins = self.n_fft//2+1
        self.trim = self.n_fft//2
        self.chunk_size = self.hop * (self.dim_t-1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=False).to(self.device)
        self.freq_pad = torch.zeros([1, self.dim_c, self.n_bins-self.dim_f, self.dim_t]).to(self.device)
        self.gen_size = self.chunk_size-2*self.trim

    def initialize_mix(self, mix, is_ckpt=False):
        if is_ckpt:
            pad = self.gen_size + self.trim - ((mix.shape[-1]) % self.gen_size)
            mixture = np.concatenate((np.zeros((2, self.trim), dtype='float32'),mix, np.zeros((2, pad), dtype='float32')), 1)
            num_chunks = mixture.shape[-1] // self.gen_size
            mix_waves = [mixture[:, i * self.gen_size: i * self.gen_size + self.chunk_size] for i in range(num_chunks)]
        else:
            mix_waves = []
            n_sample = mix.shape[1]
            pad = self.gen_size - n_sample%self.gen_size
            mix_p = np.concatenate((np.zeros((2,self.trim)), mix, np.zeros((2,pad)), np.zeros((2,self.trim))), 1)
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i:i+self.chunk_size])
                mix_waves.append(waves)
                i += self.gen_size
                
        mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(self.device)

        return mix_waves, pad
    
    def demix_base(self, mix, is_ckpt=False, is_match_mix=False):
        chunked_sources = []
        for slice in mix:
            sources = []
            tar_waves_ = []
            mix_p = mix[slice]
            mix_waves, pad = self.initialize_mix(mix_p, is_ckpt=is_ckpt)
            mix_waves = mix_waves.split(1)
            pad = mix_p.shape[-1] if is_ckpt else -pad
            with torch.no_grad():
                for mix_wave in mix_waves:
                    tar_waves = self.run_model(mix_wave, is_ckpt=is_ckpt, is_match_mix=is_match_mix)
                    tar_waves_.append(tar_waves)
                tar_waves_ = np.vstack(tar_waves_)[:, :, self.trim:-self.trim] if is_ckpt else tar_waves_
                tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :pad]
                start = 0 if slice == 0 else self.margin
                end = None if slice == list(mix.keys())[::-1][0] or self.margin == 0 else -self.margin
                sources.append(tar_waves[:,start:end]*(1/self.adjust))
            chunked_sources.append(sources)
        sources = np.concatenate(chunked_sources, axis=-1)
        
        return sources

    def run_model(self, mix, is_ckpt=False, is_match_mix=False):
        
        spek = self.stft(mix.to(self.device))*self.adjust
        spek[:, :, :3, :] *= 0 

        if is_match_mix:
            spec_pred = spek.cpu().numpy()
        else:
            spec_pred = -self.model_run(-spek)*0.5+self.model_run(spek)*0.5 if self.is_denoise else self.model_run(spek)

        if is_ckpt:
            return self.istft(spec_pred).cpu().detach().numpy()
        else: 
            return self.istft(torch.tensor(spec_pred).to(self.device)).to(cpu)[:,:,self.trim:-self.trim].transpose(0,1).reshape(2, -1).numpy()
    
    def stft(self, x):
        x = x.reshape([-1, self.chunk_size])
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True,return_complex=True)
        x=torch.view_as_real(x)
        x = x.permute([0,3,1,2])
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,self.dim_c,self.n_bins,self.dim_t])
        return x[:,:,:self.dim_f]

    def istft(self, x, freq_pad=None):
        freq_pad = self.freq_pad.repeat([x.shape[0],1,1,1]) if freq_pad is None else freq_pad
        x = torch.cat([x, freq_pad], -2)
        x = x.reshape([-1,2,2,self.n_bins,self.dim_t]).reshape([-1,2,self.n_bins,self.dim_t])
        x = x.permute([0,2,3,1])
        x=x.contiguous()
        x=torch.view_as_complex(x)
        x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
        return x.reshape([-1,2,self.chunk_size])

def prepare_mix(mix, chunk_set, margin_set, mdx_net_cut=False, is_missing_mix=False):
    samplerate = 44100

    if not isinstance(mix, np.ndarray):
        mix, samplerate = librosa.load(mix, mono=False, sr=44100)
    else:
        mix = mix.T

    if mix.ndim == 1:
        mix = np.asfortranarray([mix,mix])

    def get_segmented_mix(chunk_set=chunk_set):
        segmented_mix = {}
        
        samples = mix.shape[-1]
        margin = margin_set
        chunk_size = chunk_set*44100
        assert not margin == 0, 'margin cannot be zero!'
        
        if margin > chunk_size:
            margin = chunk_size
        if chunk_set == 0 or samples < chunk_size:
            chunk_size = samples
        
        counter = -1
        for skip in range(0, samples, chunk_size):
            counter+=1
            s_margin = 0 if counter == 0 else margin
            end = min(skip+chunk_size+margin, samples)
            start = skip-s_margin
            segmented_mix[skip] = mix[:,start:end].copy()
            if end == samples:
                break
            
        return segmented_mix

    if is_missing_mix:
        return mix
    else:
        segmented_mix = get_segmented_mix()
        raw_mix = get_segmented_mix(chunk_set=0) if mdx_net_cut else mix
        return segmented_mix, raw_mix, samplerate

if len(sys.argv) < 2:
    print_with_timestamp("Please provide the WAV audio file path to separate as the first command-line parameter.")
    sys.exit(1)

separator = Separator(sys.argv[1])
separator.separate()

print_with_timestamp("Separation complete!")
