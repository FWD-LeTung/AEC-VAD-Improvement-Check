
import numpy as np
import torch
import librosa
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from utils.ERLE import calculate_erle_series
from utils.ERLE import calculate_convergence_time
from pyaec import Aec
from silero_vad import read_audio, load_silero_vad, get_speech_timestamps
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from LinearAlgorithm.time_domain_adaptive_filters.lms import lms
from LinearAlgorithm.time_domain_adaptive_filters.blms import blms
from LinearAlgorithm.time_domain_adaptive_filters.nlms import nlms
from LinearAlgorithm.time_domain_adaptive_filters.bnlms import bnlms
from LinearAlgorithm.time_domain_adaptive_filters.kalman import kalman

from model import architecture as arc

model_path = "../model/aec_7_percent_loss/aec_v2_step_2900.pth"
mic_path = "../audio/nearend_mic_fileid_0.wav"
ref_path = "../audio/farend_speech_fileid_0.wav"
clean_path = "../audio/nearend_speech_fileid_0.wav"
neuralAEC_est = "../audio/neural_aec_est.wav"

arc.run_inference(model_path, mic_path, ref_path, neuralAEC_est)

x, sr = librosa.load(mic_path, sr=None)
r, _ = librosa.load(ref_path, sr=None)
c, _ = librosa.load(clean_path, sr=None)
n_aec, _ = librosa.load(neuralAEC_est, sr=None)


outputs = {
    "clean": c,
    "Neural AEC": n_aec
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
colors = ["#a0e61f", "#26d8d8", "#d644e9", "#2926e2", "#52d829", "#ffa43d", "#f12525"]

gt_mask, gt_ts = get_vad_mask(c)
actual_start_time = (gt_ts[0]['start']/sr)*1000 if gt_ts else 0

results = {}
# Tạo 2 đồ thị riêng biệt (Subplots) chung trục X thời gian
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
colors = ["#a0e61f", "#26d8d8", "#d644e9", "#2926e2", "#52d829", "#ffa43d", "#f12525"]

for i, (name, audio) in enumerate(outputs.items()):
    # Chạy VAD
    mask, ts = get_vad_mask(audio)
    
    # Tính F1, FPR, FNR (Nhóm A)
    min_len = min(len(gt_mask), len(mask))
    y_true, y_pred = gt_mask[:min_len], mask[:min_len]
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # Tính Delay (Nhóm B)
    first_detect_ms = (ts[0]['start']/sr)*1000 if ts else 0
    delay = max(0, first_detect_ms - actual_start_time) if ts else "N/A"
    
    # Tính Convergence Time
    conv_time = calculate_convergence_time(x, audio, sr=sr)
    conv_time_str = f"{conv_time:.2f}s" if conv_time is not None else ">5s"

    results[name] = {
        "F1": f1, "FPR": fpr, "FNR": fnr, 
        "Delay": delay, "Conv": conv_time_str
    }
    
    # Đồ thị 1: VAD Activation
    time_axis = np.arange(len(audio)) / sr
    ax1.fill_between(time_axis, i, i + mask, color=colors[i % len(colors)], alpha=0.7, label=f"{name}")

    # Đồ thị 2: ERLE Curves (Chỉ vẽ cho các thuật toán AEC, bỏ qua file clean)
    if name != "clean":
        erle_data = calculate_erle_series(x, audio)
        erle_time = np.linspace(0, len(audio)/sr, len(erle_data))
        ax2.plot(erle_time, erle_data, color=colors[i % len(colors)], label=f"{name} ERLE", linewidth=1.5)

# 6. In bảng kết quả
print(f"\n{'Thuật toán':<15} | {'F1':<6} | {'FPR':<6} | {'FNR':<6} | {'Delay(ms)':<10} | {'Conv Time'}")
print("-" * 75)
for name, m in results.items():
    delay_str = f"{m['Delay']:.2f}" if isinstance(m['Delay'], float) else m['Delay']
    print(f"{name:<15} | {m['F1']:6.2f} | {m['FPR']:6.2f} | {m['FNR']:6.2f} | {delay_str:<10} | {m['Conv']}")