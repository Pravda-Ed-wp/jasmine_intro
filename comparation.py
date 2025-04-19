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
from PIL import Image

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

   
    &emsp;&emsp;调式（Mode）是构成乐曲的若干个乐音按照一定的关系组合成的体系，通常以其中的一个音为主音，其余各音都倾向于它。在作曲时，往往将调式主音作为乐曲的起始与终止音，尤其是终止音，需要通过“终止式”将乐曲由不稳定的状态进行“解决”，给人在听觉上以结束感。一首曲子的听感很大程度上由调式决定，大体上大调给人以明亮欢快的感觉，而小调给人以宁静悲伤之感。

    中国民族音乐调式极具特色，区别于西方的“自然大调”、“自然小调”，汉民族及绝大部分少数民族民歌使用的调式都被称为“五声调式”。五声调式以纯五度音程关系排列，由宫（C）、商（D）、角（E）、徵（G）、羽（A）作为调式的“正音”，具有主导性意义；而其余的音被称为“偏音”，其出现频率需要远小于正音，并不能作为乐句的主音与重音而出现。

    五声调式中，五个主音皆可作为调式主音（自然大小调式分别以C与A作为调式主音）分别被称为宫调式、商调式、角调式、徵调式、羽调式。其中，宫调式与自然大调听感相似，羽调式与自然小调听感相似；徵调式是最具有中国特色的调式，由于使用音阶的不同，听感上既可以类似于大调（如《茉莉花》》），也可以类似于小调（《沂蒙山小调》）；商调式与角调式由于终止式不明显，因此听感上缺乏结束感，而具有延续性。角调式出现最少，仅在华东地区有零星分布。

    
    （2）旋律曲折度

    
    &emsp;&emsp;在欣赏音乐时，“婉转曲折”是听众经常用于描述民歌的特征。为了对这一特征进行定量性的描述，我们设计了一种计算方式，将这种定性描述转换为定量指标。

    
    （3）旋律协和度
    

    &emsp;&emsp;音乐风格同样会受到相邻音音程关系的影响。同样作为典型的单声部民歌，日本民歌与传统音乐中常用的“都节调式”（3 4 6 7 1）相较于五声调式，其中以小二度（3 4；7 1，一个半音）与增四度、减五度（4 7，六个半音）为代表的不协和音程常常给人以阴郁悲伤之感；而五声调式（1 2 3 5 6）极大地排除了不协和音程，从而在整体上给人以协调、明亮的听感。为了对音程关系带来的听感进行分析，我们设计了旋律协和度指标。

    
    &emsp;&emsp;最终得到的结果如下：

    """
    st.markdown(text)

    input_csv = "分析结果.csv"
    music_df=pd.read_csv(input_csv,encoding='utf-8')

    st.dataframe(music_df)

     
    
    text="""
    &emsp;&emsp;华北、东北的版本，如山西临汾、辽宁东港、吉林梨树、河北昌黎等，使用的都是在五声音阶基础上发展出的六声、七声音阶；而浙江、江苏、福建等版本大多使用的是纯五声音阶。这一现象是符合中国民歌的规律的，一般来说，江浙、江淮、荆楚、西南的民歌使用的音阶中，五声音阶占据了绝对主导地位，而东北、西北、华北中原等地的六声音阶、清乐音阶、燕乐音阶使用比例更高。《茉莉花》不同版本使用的音阶变化体现出了同一音乐题材在传播过程中受到当地音乐特色的影响，并分化出了不同版本，体现出民乐丰富多彩的特征。
    
    &emsp;&emsp;相比之下，不同版本的《茉莉花》使用的调式变化没有音阶那么丰富。由于江苏版与河北版《茉莉花》都是徵调式，因此绝大多数版本都沿用了这一调式。值得注意的是，黑龙江齐齐哈尔市的一个版本使用了羽调式，河北省滦县版本使用了商调式。在乔建中先生的著作《音地关系》中，东北部平原色彩区（包括了华北、东北平原）的民歌以徵调式为主，其次是商调式；羽调式在这一区域的使用是很少的，但是齐齐哈尔市位于黑龙江省与内蒙古自治区的交界处，而在蒙古族、满族等少数民族民歌中，羽调式运用非常广泛，因此可以推测当地的民歌收到了一定少数民族民歌的影响。

    &emsp;&emsp;江苏版《茉莉花》的平均单音长度普遍低于河北版，且大体上呈现随纬度升高单音长度增加的特征；此外，河北版《茉莉花》由于连续重复单音比例与大三度旋律进行较多，因此协和度高于以小三度级进的江苏版[9]；山西临汾版《茉莉花》非常具有西北民歌特色，听感上有一定的悲伤感，而在协和度分析中，这一版本分值最低，体现出地方特色。
    """
    st.markdown(text)


    text="""
     &emsp;&emsp;我们进一步地将这乐理分析结果结果与自然环境指标进行相关性分析，得到结果如下：
    """
    band = Image.open('《茉莉花》乐理特征与地理要素相关系数热力图.png')
    st.image(band, caption='《茉莉花》乐理特征与地理要素相关系数热力图')
    
    text="""
    &emsp;&emsp;我们初步选择了经度、纬度、河网密度与地形起伏度四个指标与乐理指标之间进行了相关性分析。在α=0.005的显著性水平之下，节奏变化与曲折程度指标与河网密度之间呈现出显著的相关性，而协和度指标与地形起伏程度之间呈现显著负相关。我们猜测，河网密度较高的区域音乐节奏变化相对更多，可能与当地经济活动活跃，歌曲形式多样有关；此外，这些区域的旋律曲折程度高，可能与当地航运业发达，“船歌”、“渔歌”等歌曲形式有关。地形起伏度与旋律协和度呈现复相关，体现出山地丘陵区域民歌偏小调的听感。
    
    &emsp;&emsp;由此可见，民族音乐的乐理特征确实可能受到了自然环境的显著影响，也希望能够更多研究者从这一角度，思考我们的音乐文化，甚至是方言声调，在形成过程中究竟是如何生于大地，长于大地的。
    """
    
        
    st.markdown(text)
    



    
