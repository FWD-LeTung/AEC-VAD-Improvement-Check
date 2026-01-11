
import numpy as np
import torch
import librosa
import sys
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from utils.ERLE import calculate_erle_series
from utils.ERLE import calculate_convergence_time
from pyaec import Aec
from silero_vad import read_audio, load_silero_vad, get_speech_timestamps
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LinearAlgorithm.time_domain_adaptive_filters.lms import lms
from LinearAlgorithm.time_domain_adaptive_filters.blms import blms
from LinearAlgorithm.time_domain_adaptive_filters.nlms import nlms
from LinearAlgorithm.time_domain_adaptive_filters.bnlms import bnlms
from LinearAlgorithm.time_domain_adaptive_filters.kalman import kalman

from model import architecture as arc

model_path = "../model/aec_v30_12/aec_cp_6500.pth"
mic_path = "../audio/160903_mic.wav"
ref_path = "../audio/160903_ref.wav"
clean_path = "../audio/160903_clean.wav"
neuralAEC_est = f"../audio/{Path(clean_path).stem}neural_aec_est.wav"
pyaec_out_path = f"../audio/{Path(clean_path).stem}pyaec_output.wav"
arc.run_inference2(model_path, mic_path, ref_path, neuralAEC_est)

frame_size = 160
filter_length = 1600
sample_rate = 16000
aec = Aec(frame_size, filter_length, sample_rate, False)

rec_samples, _ = sf.read(mic_path, dtype='int16')
ref_samples, _ = sf.read(ref_path, dtype='int16')

num_frames = len(rec_samples)//frame_size
output_frames = []

for i in range(num_frames):
    start = i * frame_size
    end = start + frame_size
    processed_frame = aec.cancel_echo(rec_samples[start:end], ref_samples[start:end])
    output_frames.append(processed_frame)

output = np.concatenate(output_frames, dtype="int16")
sf.write(pyaec_out_path, output, sample_rate)
print(f"Created {pyaec_out_path}")

x, sr = librosa.load(mic_path, sr=None)
r, _ = librosa.load(ref_path, sr=None)
c, _ = librosa.load(clean_path, sr=None)
n_aec, _ = librosa.load(neuralAEC_est, sr=None)
pyaec, _ = librosa.load(pyaec_out_path, sr=None)

outputs = {
    "clean": c,
    "Neural AEC": n_aec,
    "PyAEC": pyaec
}



model_vad = load_silero_vad()
def get_vad_mask(audio, sr=sr):
    # FIX LỖI: Ép kiểu về .float() (float32) trước khi đưa vào mô hình VAD
    audio_tensor = torch.from_numpy(audio).float() 
    timestamps = get_speech_timestamps(audio_tensor, model_vad, sampling_rate=sr)
    mask = np.zeros(len(audio))
    for ts in timestamps:
        mask[ts['start']:ts['end']] = 1
    return mask, timestamps

gt_mask, gt_ts = get_vad_mask(c)
actual_start_time = (gt_ts[0]['start']/sr)*1000 if gt_ts else 0

results = {}
plt.figure(figsize=(12, 12))
colors = ["#a0e61f", "#26d8d8", "#ffa43d"]

gt_mask, gt_ts = get_vad_mask(c)
actual_start_time = (gt_ts[0]['start']/sr)*1000 if gt_ts else 0

results = {}

for i, (name, audio) in enumerate(outputs.items()):
    # Chạy VAD
    mask, ts = get_vad_mask(audio)
    
    # Tính F1, FPR, FNR (Nhóm A)
    min_len = min(len(gt_mask), len(mask))
    y_true, y_pred = gt_mask[:min_len], mask[:min_len]
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    Precision = tp / (tp + fp) if (tp+fp) > 0 else 1
    Recall = tp / (tp + fn) if (tp + fn) > 0 else 1
    
    # Tính Delay (Nhóm B)
    first_detect_ms = (ts[0]['start']/sr)*1000 if ts else 0
    delay = max(0, first_detect_ms - actual_start_time) if ts else "N/A"
    
    # Tính Convergence Time
    conv_time = calculate_convergence_time(x, audio, sr=sr)
    conv_time_str = f"{conv_time:.2f}s" if conv_time is not None else ">5s"

    results[name] = {
        "F1": f1, "FPR": Precision, "FNR": Recall, 
        "Delay": delay, "Conv": conv_time_str
    }
    time_axis = np.arange(len(audio)) / sr
    plt.fill_between(time_axis, i, i + mask, color=colors[i % len(colors)], alpha=0.7, label=name)
    
plt.yticks(range(len(outputs)), outputs.keys())
plt.xlabel("Thời gian (giây)")
plt.title("So sánh VAD Activation giữa Clean, Neural AEC và PyAEC")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("VAD_Comparison_Plot.png")

# 6. In bảng kết quả
print(f"\n{'Thuật toán':<15} | {'F1':<6} | {'Precision':<6} | {'Recall':<6} | {'Delay(ms)':<10} | {'Conv Time'}")
print("-" * 75)
for name, m in results.items():
    delay_str = f"{m['Delay']:.2f}" if isinstance(m['Delay'], float) else m['Delay']
    print(f"{name:<15} | {m['F1']:6.2f} | {m['FPR']:6.2f} | {m['FNR']:6.2f} | {delay_str:<10} | {m['Conv']}")