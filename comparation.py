# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:48:56 2024

@author: 15311
"""
import streamlit as st
import librosa
import numpy as np
import pandas as pd
from scipy.signal import correlate
import matplotlib.pyplot as plt
def find_time_shift(audio1, audio2):
    # 计算两个音频信号的交叉相关
    cross_corr = correlate(audio1, audio2, mode='full')
    # 找到交叉相关的峰值位置
    shift = np.argmax(cross_corr) - len(audio1) + 1

    return shift

def align_audio(audio1, audio2, shift):
    # 如果需要，对音频2进行时移以对齐音频1
    if shift > 0:
        audio2_aligned = np.pad(audio2, (shift, 0), mode='constant')[:len(audio1)]
    elif shift < 0:
        audio2_aligned = audio2[-shift:][:len(audio1)]
    return audio2_aligned

def MFCC_sim(audio1, audio2, sr1, sr2):
    # 确保采样率一致，如果不一致，只调整音频2的采样率
    if sr1 != sr2:
        audio2 = librosa.resample(audio2, sr2, sr1)
    
    # 计算时移
    shift = find_time_shift(audio1, audio2)
    
    # 对齐音频2
    audio2_aligned = align_audio(audio1, audio2, shift)
    
    # 截取音频到最短长度
    min_length = min(len(audio1), len(audio2_aligned))
    audio1 = audio1[:min_length]
    audio2_aligned = audio2_aligned[:min_length]
    
    # 计算MFCC特征
    mfcc_a = librosa.feature.mfcc(y=audio1, sr=sr1)
    mfcc_b = librosa.feature.mfcc(y=audio2_aligned, sr=sr1)
    
    try:
        cos_sim = np.dot(mfcc_a, mfcc_b.T) / (np.linalg.norm(mfcc_a, axis=1) * np.linalg.norm(mfcc_b, axis=1))
        cos_sim_mean = np.mean(cos_sim)
    except Exception as e:
        cos_sim_mean = 0  # 在出现异常时返回默认相似度

    return cos_sim_mean
    print(f"两段音乐的余弦相似度为: {cos_sim_mean}")
    
def MFCC_show(audio,sr):
    # 提取MFCC特征
    mfccs = librosa.feature.mfcc(y=audio, sr=sr)
    # 可视化MFCC特征
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, x_axis='time', sr=sr, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='MFCC')
    st.pyplot(fig)

def music_node_show(audio,sr):
    #提取音高
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, S=None, n_fft=4096, hop_length=None, fmin=27.5, fmax=4186.0, 
                                           threshold=0.9999, win_length=None, window='hann', center=True, pad_mode='constant', ref=None)
    #print(pitches.shape)
    df=pd.DataFrame(pitches)
    #print(df.shape)
    #音高转化为音符
    df=df.applymap(lambda x: librosa.hz_to_note(x) if x>0 else 0)
     #生成7个八度的index_list
    list0 =["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"]
    index_list=[]
    for i in range(1,8):
        index_list=index_list+[ x+str(i) for x in list0]
    print(index_list)
    #音符转化为显示序列
    df=df.applymap(lambda x: index_list.index(x)+1 if x in index_list else 0)
    #print(df)
    #df.to_excel("D://大学//地理//大创 音乐地理//index.xlsx", encoding='gbk')
    #提取音符显示序列
    ones= np.ones(df.shape[0])
    #print(ones.shape)
    note_values=ones.dot(df)
    #print(note_values)
    #展示音高与时长图谱
    plt.figure(figsize=(80,15))
    plt.xlabel('time')
    plt.ylabel('notes')
    plt.grid(linewidth=0.5,alpha=0.5)
    plt.xticks(range(0,df.shape[1],20))
    plt.yticks(range(1,len(index_list)+1),index_list)
    plt.plot(note_values,color="#008080",linewidth=0.8)
    plt.hlines(note_values, np.array(range(len(note_values)))-0.5,np.array(range(len(note_values)))+0.5,color="red", linewidth=5)
    st.pyplot(plt)
    
def run():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    st.title('音乐相似度分析')
    text="""
    &emsp;&emsp;那么，我们最熟悉的《茉莉花》是上述哪一种的茉莉花呢？实际上，如果我们去研究，会发现我们听到的版本和《图兰朵》版本更为接近。而且，该版本的出现时间大多在00~10年代，在更早的时期更多演唱版本都是江苏版。我们为什么会对这一“舶来品”认可度如此之高呢？我将其与若干本土版本进行了相似度比较。
    """
    st.markdown(text)
    filepaths = [
        "图兰朵茉莉花.mp3",
        "江苏茉莉花.mp3",
        "东北茉莉花.mp3",
        "河南茉莉花.mp3",
        "山西茉莉花.mp3"
    ]
    main_audio, main_sr = librosa.load(filepaths[0])
    music_node_show(main_audio,main_sr)
    MFCC_show(main_audio, main_sr)
    text="""
    &emsp;&emsp;上面二图分别展示了《图兰朵》版茉莉花的音高与频谱特征。进一步使用其MFCC特征与其他几个版本进行比较，可得到相似度如下：
    """
    st.markdown(text)
    similarities = []
    labels = []
    for filepath in filepaths[1:]:
        audio, sr = librosa.load(filepath)
        similarity = MFCC_sim(main_audio, audio, main_sr, sr)
        similarities.append(similarity)
        labels.append(filepath.split('/')[-1].split('.')[0])
    plt.figure(figsize=(10, 6))
    plt.bar(labels, similarities, color='blue')
    plt.xlabel('音频文件')
    plt.ylabel('相似度')
    plt.title('新版《茉莉花》的相似度分析')
    st.pyplot(plt)
    text="""
    &emsp;&emsp;从图中可以看出，新版《茉莉花》的特征处于几个主要版本《茉莉花》的当中，处于南派与北派茉莉花的中间。江浙地区民歌优雅、婉转、清丽，但是音阶以五声音阶作为绝对主导，旋律线受到 “三音”一组的级进型乐汇支配，少跳跃；北方民歌旋律较为朴实简单，活泼欢快，但音阶更为复杂，多以六声、七声音阶为主。新版《茉莉花》在听感上兼有南方五声音阶的民族性，又有北方民歌旋律线的朴实简单，同时首句的重复与后续旋律走向带有对称、稳健的特征，使得在听感上很容易被南北方的听众接受。通过同时削减两方面的特征达到折中，并最大限度的保留民族特征，才是新版《茉莉花》被广泛接受的根本原因。
    """
    st.markdown(text)

    
        
    
    



    
