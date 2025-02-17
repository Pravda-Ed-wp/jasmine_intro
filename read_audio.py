# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 10:00:53 2024

@author: 15311
"""

import os
import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def extract_audio(direction="./音乐数据/audio_files"):
    audio_dir=direction
    audio_files = []
    for root, dirs, files in os.walk(audio_dir):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                audio_files.append(os.path.join(root, file))

    # 提取特征并保存结果
    data = []
    for file_path in audio_files:
        try:
            print(f"Processing file: {file_path}")
            features = extract_features(file_path)
            file_name = os.path.splitext(os.path.basename(file_path))[0]
            data.append([file_name] + features.tolist())
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            
    # 转换为DataFrame
    columns = ['file_name'] + [f'mfcc_{i}' for i in range(13)]
    df = pd.DataFrame(data, columns=columns)
    
    # 保存到CSV文件
    output_csv = "./音乐数据/audio_features.csv"
    df.to_csv(output_csv, index=False,encoding='gbk')
    
    print(f"特征提取完毕，结果已保存到 {output_csv}")
    
extract_audio()


