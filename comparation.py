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
    &emsp;&emsp;在第一和二部分中，我们已经感受到《茉莉花》的不同版本之间存在一定的差异。然而，是什么因素导致了这样的差异呢？可能大家第一时间想到的，就会是不同地区的自然环境，以及这一因素导致生产生活方式的不同。
    这个想法究竟对不对呢？我们对此进行了一定的分析，试图找寻乐理特征的形成究竟是受到了哪些自然要素的影响。
    """
    st.markdown(text)
    filepaths = [
        "图兰朵茉莉花.mp3",
        "江苏茉莉花.mp3",
        "东北茉莉花.mp3",
        "河南茉莉花.mp3",
        "山西茉莉花.mp3"
    ]
    #main_audio, main_sr = librosa.load(filepaths[0])
    #music_node_show(main_audio,main_sr)
    #MFCC_show(main_audio, main_sr)

     text="""
    &emsp;&emsp;我们选择进行分析的指标有以下这些：

    （1）调式

    调式（Mode）是构成乐曲的若干个乐音按照一定的关系组合成的体系，通常以其中的一个音为主音，其余各音都倾向于它。在作曲时，往往将调式主音作为乐曲的起始与终止音，尤其是终止音，需要通过“终止式”将乐曲由不稳定的状态进行“解决”，给人在听觉上以结束感。一首曲子的听感很大程度上由调式决定，大体上大调给人以明亮欢快的感觉，而小调给人以宁静悲伤之感。

    中国民族音乐调式极具特色，区别于西方的“自然大调”、“自然小调”，汉民族及绝大部分少数民族民歌使用的调式都被称为“五声调式”。五声调式以纯五度音程关系排列，由宫（C）、商（D）、角（E）、徵（G）、羽（A）作为调式的“正音”，具有主导性意义；而其余的音被称为“偏音”，其出现频率需要远小于正音，并不能作为乐句的主音与重音而出现。

    五声调式中，五个主音皆可作为调式主音（自然大小调式分别以C与A作为调式主音）分别被称为宫调式、商调式、角调式、徵调式、羽调式。其中，宫调式与自然大调听感相似，羽调式与自然小调听感相似；徵调式是最具有中国特色的调式，由于使用音阶的不同，听感上既可以类似于大调（如《茉莉花》》），也可以类似于小调（《沂蒙山小调》）；商调式与角调式由于终止式不明显，因此听感上缺乏结束感，而具有延续性。角调式出现最少，仅在华东地区有零星分布。

    （2）旋律曲折度

    在欣赏音乐时，“婉转曲折”是听众经常用于描述民歌的特征。为了对这一特征进行定量性的描述，我们设计了一种计算方式，将这种定性描述转换为定量指标。

    （3）旋律协和度

    音乐风格同样会受到相邻音音程关系的影响。同样作为典型的单声部民歌，日本民歌与传统音乐中常用的“都节调式”（3 4 6 7 1）相较于五声调式，其中以小二度（3 4；7 1，一个半音）与增四度、减五度（4 7，六个半音）为代表的不协和音程常常给人以阴郁悲伤之感；而五声调式（1 2 3 5 6）极大地排除了不协和音程，从而在整体上给人以协调、明亮的听感。为了对音程关系带来的听感进行分析，我们设计了旋律协和度指标。

    最终得到的结果如下：


    
    """
    st.markdown(text)


    band = Image.open('《茉莉花》乐理特征与地理要素相关系数热力图.png')
    st.image(band, caption='《茉莉花》乐理特征与地理要素相关系数热力图')
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

    
        
    
    



    
