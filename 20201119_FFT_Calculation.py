import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 

def FFT_main(t, x, dt, split_t_r, overlap,window_F, output_FN, y_label, y_unit):
    
    # データをオーバーラップして分割
    split_data = data_split(t, x, split_t_r, overlap)
    
    # Do FFT
    FFT_result_list = []
    for split_data_cont in split_data:
        FFT_result_cont = FFT(split_data_cont, dt, window_F)
        FFT_result_list.append(FFT_result_cont)
    
    # Graphing each frame
#     IDN = 0
#     for split_data_cont, FFT_result_cont, in zip(split_data, FFT_result_list):
#         IDN = IDN + 1
#         plot_FFT(split_data_cont[0], split_data_cont[1], FFT_result_cont[0], FFT_result_cont[1], 
#                  output_FN, IDN, 0, y_label, y_unit)
        
    # Averaging
    fq_avg = FFT_result_list[0][0]
    F_abs_amp_avg = np.zeros(len(fq_avg))
    for i in range(len(FFT_result_list)):
        F_abs_amp_avg = F_abs_amp_avg + FFT_result_list[i][1]
    F_abs_amp_avg = F_abs_amp_avg / (i + 1)
    
    plot_FFT(t, x, fq_avg, F_abs_amp_avg, output_FN, 'avg', 1, y_label, y_unit)
    
    return fq_avg, F_abs_amp_avg


def plot_FFT(t, x, fq, F_abs_amp, output_FN, IDN, final_graph, y_label, y_unit):
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    title1 = 'time_' + output_FN[:-4]
    plt.plot(t, x)
    plt.xlabel('time (s)')
    plt.ylabel('y_label' + '['+ y_unit + ']')
    plt.title(title1)
    
    ax2 = fig.add_subplot(212)
    title2 = 'freq_' + output_FN[:-4]
    plt.xlabel('Frequency (Hz)')
    plt.ylabel(y_label + '[' + y_unit + '/rtHz]')
    plt.plot(fq, F_abs_amp)
    plt.title(title2)
    
    
def FFT(data_input, dt, window_F):
    N = len(data_input[0])
    
    # Window Funcion
    if window_F == 'hanning':
        window = np.hanning(N)
    elif window_F == 'hamming':
        window = np.hamming(N)
    elif window_F == 'blackman':
        window = np.blackman(N)
    else:
        print('Error: input window function name is not supported')
        
    # 窓関数後の信号
    x_windowed = data_input[1]*window
    
    # FET Calculation
    F = np.fft.fft(x_windowed)
    F_abs = np.abs(F)
    F_abs_amp = F_abs / N * 2
    fq = np.linspace(0, 1.0/dt, N)
    
    # 窓関数の補正値
    acf = 1 / (sum(window) / N)
    F_abs_amp_out = F_abs_amp[:int(N/2) + 1]
    
    # ナイキスト定数まで抽出
    fq_out = fq[:int(N/2) + 1]
    
    return [fq_out, F_abs_amp_out]


split_data = []
def data_split(t, x, split_t_r, overlap):
    # split_data = []
    one_frame_N = int(len(t)*split_t_r) # samples / frame
    overlap_N = int(one_frame_N*overlap) # overlapped samples
    start_S = 0 # サンプリングスタート位置
    end_S = start_S + one_frame_N # サンプリングエンド位置
    
    while True:
        t_cont = t[start_S:end_S] # Sampling for t
        x_cont = x[start_S:end_S] # Sampling fot x
        split_data.append([t_cont, x_cont])
        
        start_S = start_S + (one_frame_N - overlap_N)
        end_S = start_S + one_frame_N
        
        if end_S > len(t):
            break
            
        return np.array(split_data)
    


if __name__ == '__main__':
    data = pd.read_csv('sample_data_for_FFT.csv')
    t = data.iloc[:, 0]
    x = data.iloc[:, 1]
    
    print(data.head())
    
    dt = 0.01 # サンプリング周期
    output_FN = 'test.png'
    
    split_t_r = 0.1 # 窓の範囲
    overlap = 0.5 # 窓関数のオーバーラップ
    window_F = 'hanning'
    y_label = 'amplitude'
    y_unit = 'V'
    FFT_main(t, x, dt, split_t_r, overlap, window_F, output_FN, y_label, y_unit)
    
    