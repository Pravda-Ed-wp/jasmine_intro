# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 18:26:33 2024

@author: 15311
"""

import base64
import streamlit as st
from PIL import Image

def main_bg(main_bg):
    main_bg_ext = "png"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )
 
#调用
#main_bg('./pics/background.jpg')
def run():
    band = Image.open('./pics/4364b2e3f09059753188c4a3d2f79399.jpeg')

    st.image(band, caption='清朝时期的民乐团')

    text="&emsp;&emsp;说到《茉莉花》，大家第一想到的是什么？我想，最先想到的应该是这首歌："
    st.markdown(text)
    audio_file1 = open('./audio/图兰朵茉莉花.mp3', 'rb')
    audio_bytes = audio_file1.read()
    st.audio(audio_bytes, format='audio/ogg')

    text="&emsp;&emsp;或许有人会提出异议，认为这是原版的《茉莉花》："
    st.markdown(text)

    audio_file2 = open('./audio/江苏茉莉花.mp3', 'rb')
    audio_bytes = audio_file2.read()
    st.audio(audio_bytes, format='audio/ogg')

    text="&emsp;&emsp;但是我想，把茉莉花和这首歌联系起来的人应该少得多："
    st.markdown(text)

    audio_file3 = open('./audio/东北茉莉花.mp3', 'rb')
    audio_bytes = audio_file3.read()
    st.audio(audio_bytes, format='audio/ogg')


    text="""
    &emsp;&emsp;没错，在《茉莉花》传播的过程当中，在天南地北产生了很大的分异现象。尤其是在北方，山西、辽宁，
    都发展出了具有显著地方特色的《茉莉花》版本。\n\n&emsp;《茉莉花》是一首江苏民歌。相关记录最早见于清朝乾隆
    年间戏曲剧本集《缀白裘》中的《花鼓曲》。但由于其中只有“奴要摘一朵花来戴”一句歌词与今日之《茉莉花》有关，
    因此难以确定为《茉莉花》的初始版本。后有人选取《花鼓曲》中不同段落的歌词，编为《鲜花调》和《茉莉花》。（《鲜花调》和《茉莉花》被认为是同宗民歌。）
    最早的曲谱则见于清道光元年一本小百科知识全书《小慧集》。这本小书中收录了《鲜花调》的工尺谱。

   
    “我从未见过有人能像那个中国人那样唱歌，歌声充满了感情而直白。
    他在一种类似吉他的乐器伴奏下，唱了这首赞美茉莉花的歌。”"""
    st.markdown(text)
    text="""
    &emsp;&emsp;上面这句话出自1804年出版的《中国旅行记》。作者约翰·巴罗在1792-1794年间任英国首任驻华大使马戛尔尼伯爵的秘书。
    在其卸任返回英国的途中，巴罗在广州听到了《茉莉花》一曲，并发出赞叹。从此我们也可以看出，《鲜花调》早在清朝便已经传唱大江南北，广为人知。

    &emsp;&emsp;为方便歌曲在欧洲传唱，巴罗以五线谱记录下曲谱。使团听事官西特纳为这一曲调加上了引子、尾声、伴奏等并发布了出来。实际上，他们的传播是非常成功的，在19世纪，各个
    介绍中国的音乐教科书，或者是旅行记中，都很容易能够看到《茉莉花》的身影。但真正让《茉莉花》走向世界的，是1924年
    意大利著名歌剧家G.普契尼在歌剧《图兰朵》中，引用了这一首中国民歌，写在了合唱曲《当月亮升起的时候》中。据说，这是
    受到了他朋友送给他的一个八音盒的启发而写出的。

    """
    st.markdown(text)
    video_file = open('图兰朵里的茉莉花 - 1.图兰朵(Turandot.1988).歌剧.普契尼(Puccini).多明戈(Av66546311,P1).mp4', 'rb')
    video_bytes = video_file.read()
    # 使用st.video函数播放视频
    st.video(video_bytes)
    st.markdown("""
                <p style='text-align: center; font-size: 12px; margin-top: 10px;'>
                图兰朵(Turandot.1988).作者：普契尼(Puccini).演唱者：多明戈
                </p>""", unsafe_allow_html=True)

    st.markdown("""
    &emsp;&emsp;那么，为什么我们熟悉的《茉莉花》却不像是江苏原版的茉莉花？各个地方的茉莉花之间差别又有多大？是什么引起了他们之间的差异？我们将从《茉莉花》在地理上的分析窥见一二。
    """)
#cd E:\音乐地理
#streamlit run E:\音乐地理\visualize.py