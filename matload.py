import math
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import scipy.io
import pandas as pd
import warnings

from pathlib import Path

warnings.filterwarnings("ignore")


def calculate_time(list_offsets, list_onsets):
    """The function calculates the median the median duration of the given peaks

    Parameters
    ----------
    list_offsets : list
        Beginnings of peaks on the timeline
    list_onsets : list
        Ends of peaks on the timeline
    """

    list_time = []

    # If the length of list_offsets is not equal to list_onsets, it will cause problems in further calculations.
    # In this case, the solution is to take the shorter length of one of the lists and stop calculating when
    # this number is reached.
    if len(list_offsets) != len(list_onsets):
        minimum = min(len(list_offsets), len(list_onsets))
    else:
        minimum = len(list_offsets)
    for i in range(minimum):
        if list_offsets[i] - list_onsets[i] < 0:
            if i + 1 >= minimum:
                break
            list_time.append(list_offsets[i + 1] - list_onsets[i])
        else:
            list_time.append(list_offsets[i] - list_onsets[i])

    # If list_time is not nothing then calculate and return median, otherwise
    # (something went wrong and it's not gathered) pass this chunk of code
    if list_time:
        time = np.quantile(list_time, 0.5)
        return time
    else:
        pass


def analyze_ECG(ecg, sampling_rate, method="dwt"):
    """The function returns analysis parameters that can be used for display and further analysis.
Returns heart rate, average HRV, HRVSDNN, HRVRMSSD, wave related information (length, time, amplitude).

    Parameters
    ----------
    ecg : list
        The values of the time axis and the amplitude of the ECG signal
    sampling_rate : int
        Sampling rate
    method : str, optional
        Wavelet method for feature extraction (default is "dwt")
    """

    a, peaks = nk.ecg_process(ecg[1], sampling_rate=sampling_rate)
    info = nk.ecg_analyze(a, sampling_rate=sampling_rate)
    ECG_Rate_Mean = info["ECG_Rate_Mean"][0]
    HRV_RMSSD = info["HRV_RMSSD"][0]
    HRV_MeanNN = info["HRV_MeanNN"][0]
    HRV_SDNN = info["HRV_SDNN"][0]

    _, rpeaksnk = nk.ecg_peaks(ecg[1], sampling_rate=sampling_rate)
    R_peaks = [[], []]
    for i in rpeaksnk["ECG_R_Peaks"]:
        R_peaks[0].append(ecg[0][i])
        R_peaks[1].append(ecg[1][i])

    # Delineate the ECG signal
    _, waves_peak = nk.ecg_delineate(ecg[1], rpeaksnk, sampling_rate=sampling_rate, show=False,
                                     show_type="peaks")
    if method == "cwt":
        _, waves_other = nk.ecg_delineate(ecg[1], rpeaksnk, sampling_rate=sampling_rate,
                                          method="cwt", show=False, show_type='all')
    if method == "dwt":
        _, waves_other = nk.ecg_delineate(ecg[1], rpeaksnk, sampling_rate=sampling_rate,
                                          method="dwt", show=False, show_type='all')

    # Designation of P, Q, S, T waves
    P_peaks = [[], []]
    Q_peaks = [[], []]
    S_peaks = [[], []]
    T_peaks = [[], []]
    for name in waves_peak:
        for i in waves_peak[str(name)]:
            if math.isnan(i):
                continue
            if str(name) == "ECG_P_Peaks":
                P_peaks[0].append(ecg[0][i])
                P_peaks[1].append(ecg[1][i])
            if str(name) == "ECG_Q_Peaks":
                Q_peaks[0].append(ecg[0][i])
                Q_peaks[1].append(ecg[1][i])
            if str(name) == "ECG_S_Peaks":
                S_peaks[0].append(ecg[0][i])
                S_peaks[1].append(ecg[1][i])
            if str(name) == "ECG_T_Peaks":
                T_peaks[0].append(ecg[0][i])
                T_peaks[1].append(ecg[1][i])

    # Marking the beginnings and ends of the P, Q, S, T waves
    P_onsets = [[], []]
    P_offsets = [[], []]
    R_onsets = [[], []]
    R_offsets = [[], []]
    T_onsets = [[], []]
    T_offsets = [[], []]

    for name in waves_other:
        for i in waves_other[str(name)]:
            if math.isnan(i):
                continue
            if str(name) == "ECG_P_Onsets":
                P_onsets[0].append(ecg[0][i])
                P_onsets[1].append(ecg[1][i])
            if str(name) == "ECG_P_Offsets":
                P_offsets[0].append(ecg[0][i])
                P_offsets[1].append(ecg[1][i])
            if str(name) == "ECG_R_Onsets":
                R_onsets[0].append(ecg[0][i])
                R_onsets[1].append(ecg[1][i])
            if str(name) == "ECG_R_Offsets":
                R_offsets[0].append(ecg[0][i])
                R_offsets[1].append(ecg[1][i])
            if str(name) == "ECG_T_Onsets":
                T_onsets[0].append(ecg[0][i])
                T_onsets[1].append(ecg[1][i])
            if str(name) == "ECG_T_Offsets":
                T_offsets[0].append(ecg[0][i])
                T_offsets[1].append(ecg[1][i])

    # Calculating duration of the ECG waves
    P_Time = calculate_time(P_offsets[0], P_onsets[0])
    QRS_Time = calculate_time(R_offsets[0], R_onsets[0])
    T_Time = calculate_time(T_offsets[0], T_onsets[0])

    # ECG wave amplitudes
    P_Amplitude = np.mean(P_peaks[1])
    Q_Amplitude = np.mean(Q_peaks[1])
    R_Amplitude = np.mean(R_peaks[1])
    S_Amplitude = np.mean(S_peaks[1])
    T_Amplitude = np.mean(T_peaks[1])

    # ECG intervals
    PQ_Space = calculate_time(R_onsets[0], P_onsets[0])
    QT_Space = calculate_time(T_offsets[0], R_onsets[0])

    # ECG segments
    PQ_Segment = calculate_time(R_onsets[0], P_offsets[0])
    ST_Segment = calculate_time(T_onsets[0], R_offsets[0])

    # Dictionaries with acquired informations
    data = {}
    info = {}

    data["P_peaks"] = P_peaks
    data["Q_peaks"] = Q_peaks
    data["R_peaks"] = R_peaks
    data["S_peaks"] = S_peaks
    data["T_peaks"] = T_peaks
    data["P_onsets"] = P_onsets
    data["P_offsets"] = P_offsets
    data["R_onsets"] = R_onsets
    data["R_offsets"] = R_offsets
    data["T_onsets"] = T_onsets
    data["T_offsets"] = T_offsets
    try:
        info["ECG_Rate_Mean"] = round(ECG_Rate_Mean, 4)
    except (TypeError, AttributeError):
        info["ECG_Rate_Mean"] = None
    try:
        info["HRV_MeanNN"] = round(HRV_MeanNN, 4)
    except (TypeError, AttributeError):
        info["HRV_MeanNN"] = None
    try:
        info["HRV_RMSSD"] = round(HRV_RMSSD, 4)
    except (TypeError, AttributeError):
        info["HRV_RMSSD"] = None
    try:
        info["HRV_SDNN"] = round(HRV_SDNN, 4)
    except (TypeError, AttributeError):
        info["HRV_SDNN"] = None
    try:
        info["P_Time"] = round(P_Time, 4)
    except (TypeError, AttributeError):
        info["P_Time"] = None
    try:
        info["QRS_Time"] = round(QRS_Time, 4)
    except (TypeError, AttributeError):
        info["QRS_Time"] = None
    try:
        info["T_Time"] = round(T_Time, 4)
    except (TypeError, AttributeError):
        info["T_Time"] = None
    try:
        info["P_Amplitude"] = round(P_Amplitude, 4)
    except (TypeError, AttributeError):
        info["P_Amplitude"] = None
    try:
        info["Q_Amplitude"] = round(Q_Amplitude, 4)
    except (TypeError, AttributeError):
        info["Q_Amplitude"] = None
    try:
        info["R_Amplitude"] = round(R_Amplitude, 4)
    except (TypeError, AttributeError):
        info["R_Amplitude"] = None
    try:
        info["S_Amplitude"] = round(S_Amplitude, 4)
    except (TypeError, AttributeError):
        info["S_Amplitude"] = None
    try:
        info["T_Amplitude"] = round(T_Amplitude, 4)
    except (TypeError, AttributeError):
        info["T_Amplitude"] = None
    try:
        info["PQ_Space"] = round(PQ_Space, 4)
    except (TypeError, AttributeError):
        info["PQ_Space"] = None
    try:
        info["QT_Space"] = round(QT_Space, 4)
    except (TypeError, AttributeError):
        info["QT_Space"] = None
    try:
        info["PQ_Segment"] = round(PQ_Segment, 4)
    except (TypeError, AttributeError):
        info["PQ_Segment"] = None
    try:
        info["ST_Segment"] = round(ST_Segment, 4)
    except (TypeError, AttributeError):
        info["ST_Segment"] = None

    return data, info


def show_data(ecg, data):
    """Function for displaying analyzed data

    Parameters
    ----------
    ecg : list
        The values of the time axis and the amplitude of the ECG signal
    data : dictionary
        Information about features of ecg generated by "analyzeECG()" function
    """

    plt.plot(ecg[0], ecg[1])
    plt.plot(data["R_peaks"][0], data["R_peaks"][1], "ro", label="R peaks")
    plt.plot(data["P_peaks"][0], data["P_peaks"][1], "bv", label="P peaks")
    plt.plot(data["Q_peaks"][0], data["Q_peaks"][1], "kv", label="Q peaks")
    plt.plot(data["S_peaks"][0], data["S_peaks"][1], color="lightseagreen", marker="v", label="S peaks",
             linestyle="None")
    plt.plot(data["T_peaks"][0], data["T_peaks"][1], "yv", label="T peaks")
    plt.plot(data["P_onsets"][0], data["P_onsets"][1], "b^", label="P onsets")
    plt.plot(data["P_offsets"][0], data["P_offsets"][1], "b^", label="P offsets")
    plt.plot(data["R_onsets"][0], data["R_onsets"][1], "r^", label="R onsets")
    plt.plot(data["R_offsets"][0], data["R_offsets"][1], "r^", label="R offsets")
    plt.plot(data["T_onsets"][0], data["T_onsets"][1], "y^", label="T onsets")
    plt.plot(data["T_offsets"][0], data["T_offsets"][1], "y^", label="T offsets")
    plt.xlabel('time [s]')
    plt.ylabel('voltage [mV]')
    plt.legend()
    plt.show()


def extract_features(dict, path, classification):
    """Function responsible for signal processing and feature extraction

    Parameters
    ----------
    dict : dictionary
        Dictionary for data recording
    path : str
        Path to target files
    classification : str
        The name of the class
    """
    ecg_files = Path(path).glob("*.mat")
    files = [i for i in ecg_files]
    for file in files:
        print("Extracting: ", file, "; ", files.index(file) + 1, "out of ", len(files))
        mat = scipy.io.loadmat(file)
        y = []
        x = []
        for value in mat.values():
            for i in value[0]:
                y.append(i)

        # 360 Hz
        time = 0
        for _ in y:
            x.append(time)
            time += 1 / float(360)

        # 200 adu/mV
        for i in range(len(y)):
            y[i] = y[i] / 200

        ecg = [x, y]
        try:
            data, info = analyze_ECG(ecg, 360)
            for j in range(len(info)):
                dict[list(info.keys())[j]].append(list(info.values())[j])
            dict[list(dict.keys())[-1]].append(classification)
        except(ZeroDivisionError, IndexError, ValueError):
            print("Error - sample skipped...")
            continue
    return dict


ecg_data = {"ECG_Rate_Mean": [],
            "HRV_MeanNN": [],
            "HRV_RMSSD": [],
            "HRV_SDNN": [],
            "P_Time": [],
            "QRS_Time": [],
            "T_Time": [],
            "P_Amplitude": [],
            "Q_Amplitude": [],
            "R_Amplitude": [],
            "S_Amplitude": [],
            "T_Amplitude": [],
            "PQ_Space": [],
            "QT_Space": [],
            "PQ_Segment": [],
            "ST_Segment": [],
            "Class": []}

path_list = ["C:/Users/cezar/Desktop/Studia/Inżynier/MLII/1 NSR - test",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/2 APB",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/3 AFL",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/4 AFIB - test",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/5 SVTA",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/6 WPW",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/7 PVC - test",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/8 Bigeminy",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/9 Trigeminy",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/10 VT",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/11 IVR",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/12 VFL",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/13 Fusion",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/14 LBBBB - test",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/15 RBBBB",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/16 SDHB",
             "C:/Users/cezar/Desktop/Studia/Inżynier/MLII/17 PR"]
name_list = ["NSR", "APB", "AFL", "AFIB", "SVTA", "WPW", "PVC", "Bigeminy", "Trigeminy", "VT", "IVR", "VFL", "Fusion",
             "LBBBB", "RBBBB", "SDHB", "PR"]
for path, name in zip(path_list, name_list):
    print("**********")
    print("Extracting: ", path)
    print("**********")
    ecg_data = extract_features(ecg_data, path, name)
df = pd.DataFrame(ecg_data)
print(df)
df.to_csv(r"C:\Users\cezar\Desktop\Studia\Inżynier\ecg_data_all.csv", index=False)
