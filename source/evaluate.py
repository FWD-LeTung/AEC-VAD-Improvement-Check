from pyaec import Aec
import numpy as np
import torch
import librosa
from silero_vad import read_audio, load_silero_vad, get_speech_timestamps

from LinearAlgorithm.time_domain_adaptive_filters.lms import lms
from LinearAlgorithm.time_domain_adaptive_filters.blms import blms
from LinearAlgorithm.time_domain_adaptive_filters.nlms import nlms
from LinearAlgorithm.time_domain_adaptive_filters.bnlms import bnlms
from LinearAlgorithm.time_domain_adaptive_filters.kalman import kalman

from model import architecture as arc

model_path = ""
mic_path = ""
ref_path = ""
clean_path = ""
neuralAEC_est = ""

arc.run_inference(model_path, mic_path, ref_path, neuralAEC_est)

x, sr = librosa.load(mic_path, sr=None)
r, sr = librosa.load(ref_path, sr=None)

lms_est = lms(x, r, N=256, mu=0.1)
blms_est = blms(x, r, N=256, L=4, mu=0.1)
nlms_est = nlms(x, r, N=256, mu=0.1)
bnlms_est = bnlms(x, r, N=256, L=4, mu=0.1)
kalman_est = kalman(x, r, N=256)

