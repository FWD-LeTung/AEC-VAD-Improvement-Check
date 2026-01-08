import numpy as np

def calculate_erle_series(mic, est, frame_size=160):
    min_len = min(len(mic), len(est))
    mic_aligned = mic[:min_len]
    est_aligned = est[:min_len]
    num_frames = min_len // frame_size
    erle_series = []
    for i in range(num_frames):
        frame_mic = mic_aligned[i*frame_size : (i+1)*frame_size]
        frame_est = est_aligned[i*frame_size : (i+1)*frame_size]
        power_mic = np.mean(frame_mic**2) + 1e-12
        power_est = np.mean(frame_est**2) + 1e-12
        # ERLE = 10 * log10(Năng lượng Mic / Năng lượng sai số)
        erle_series.append(10 * np.log10(power_mic / power_est))
    return np.array(erle_series)

def calculate_convergence_time(mic, output, sr=16000, threshold_db=15, frame_len=160):
    """Tính thời gian để ERLE vượt ngưỡng ổn định"""
    min_len = min(len(mic), len(output))
    mic, output = mic[:min_len], output[:min_len]
    
    num_frames = min_len // frame_len
    for i in range(num_frames):
        m_f = mic[i*frame_len : (i+1)*frame_len]
        o_f = output[i*frame_len : (i+1)*frame_len]
        p_mic = np.mean(m_f**2) + 1e-10
        p_out = np.mean(o_f**2) + 1e-10
        erle = 10 * np.log10(p_mic / p_out)
        
        if erle >= threshold_db:
            return (i * frame_len) / sr
    return None