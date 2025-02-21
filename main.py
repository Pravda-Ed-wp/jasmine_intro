# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 23:37:03 2024

@author: 15311
"""

import streamlit as st
import importlib
import visualize
import comparation

def load_module(module_name):
    try:
        return importlib.import_module(module_name)
    except ImportError:
        st.error(f"无法加载模块：{module_name}")
        return None
st.title("从鲜花调到茉莉花")


def main():
    # 设置初始页面为Home
    session_state = st.session_state
    session_state['page'] = '《茉莉花》介绍'

    # 导航栏
    page = st.sidebar.radio('导航', ['《茉莉花》介绍', "音乐近似版本区", "音乐相似度比对"])

    if page == '《茉莉花》介绍':
        content1 =load_module("introduction")
        if content1:
            content1.run()

    elif page == "音乐近似版本区":
        content1 =load_module("visualize")
        if content1:
            content1.run()
        
    elif page =="音乐相似度比对":
        content1 =load_module("comparation")
        if content1:
            content1.run()

if __name__ == '__main__':
    main()

#cd E:\音乐地理
#streamlit run E:\音乐地理\main.py
