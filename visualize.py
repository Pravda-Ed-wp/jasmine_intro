# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:42:15 2024

@author: 15311
"""

import streamlit as st
from PIL import Image
import pandas as pd
from shapely.geometry import Point,MultiPoint,Polygon
import numpy as np
import folium
import geopandas as gpd
from streamlit_folium import st_folium
import random
from shapely.affinity import scale, rotate
def get_style(x):
    # 根据你的逻辑返回样式字典
    return {
        'color': 'blue',
        'weight': 0.5,
        'opacity': 0.5
    }
def keep_right_side(value):
    # 按照'_'分割字符串，并返回最后一个元素
    return value.split('_')[-1]

def trans_point_geogson(point_list):
    multipoint = MultiPoint(point_list)
    gdf_point_1 = gpd.GeoDataFrame(geometry=[multipoint], crs="EPSG:4326")
    geo_json_1 = gdf_point_1.to_json()
    return(geo_json_1)

def mapping(cities_shp):
    min_lon, min_lat, max_lon, max_lat = cities_shp.total_bounds
    center_lat = (min_lat + max_lat) / 2+ 8.5
    center_lon = (min_lon + max_lon) / 2 
    cities = cities_shp.to_crs(epsg=4326).to_json()
    city_boundaries = folium.FeatureGroup(name='城市边界')
    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=4,
                   tiles=None)
    folium.GeoJson(data=cities,style_function=get_style).add_to(city_boundaries)
    city_boundaries.add_to(m)
    tile1 = folium.TileLayer(
            tiles='Esri.WorldImagery',
            name='Esri全球影像',
            attr='Esri全球影像')
    tile1.add_to(m)
    tile2 = folium.TileLayer(
            tiles='Gaode.Normal',
            name='高德地图',
            attr='高德地图')
    tile2.add_to(m)
    folium.plugins.MousePosition().add_to(m)
    return m

def get_minimum_bounding_rect(points):
    multipoint = MultiPoint(points)
    convex_hull = multipoint.convex_hull
    if convex_hull.geom_type == 'Polygon':
        min_rotated_rect = convex_hull.minimum_rotated_rectangle
        return min_rotated_rect
    return None

def run():
    st.title('音乐近似版本区')
    text="""&emsp;&emsp;乔建中先生在他的著作《土地与歌》中将中国的民歌分为了十个近似色彩区，分别为东北部平原民歌，西北部高原民歌，江淮民歌，江浙平原民歌，闽、台民歌，粤民歌，江汉民歌，湘民歌，赣民歌，西南高原民歌近似色彩区与客家民歌特区。受到这一思路的启发，我们对《茉莉花》的各个版本进行了聚类，希望能够得到更为精确的《茉莉花》版本近似区。
    \n\n &emsp;&emsp;梅尔频率倒谱系数（MFCC, Mel-Frequency Cepstral Coefficients）是一种广泛用于音频信号处理和语音识别的特征提取方法。MFCC通过模拟人类听觉系统的感知特性来提取音频信号的特征，它能够捕捉到音频信号中的重要信息。通过对音频信号进行预处理，将音频信号转换为频域表示，计算每一帧的傅里叶变换来获得频谱，并通过滤波与离散余弦变换，将其转换为倒谱系数，其中包含了大部分的语音特征信息。对于音乐，MFCC主要作用是可以提取音色、划分流派分类并进行情感分类。本项目使用MFCC指数对各版本《茉莉花》进行分析。
    数据来源为“中华民族音乐传承出版工程数据库”，该数据库由人民音乐出版社搭建。中华民族音乐传承出版工程是中央宣传部《中华优秀传统文化传承发展工程“十四五”重点项目规划》中提出的重点项目，提供了包括采风录制、数字修复、已出版音乐资源等一系列民乐资源，并以“地图导览”的形式整合，提供了丰富的民族音乐资源，覆盖中国各省份，为研究提供了宝贵的原始材料。数据选用“地图导航”模块中搜索词为《茉莉花》、《鲜花调》与《双叠翠》的所有音频数据，并去除重复部分。
    """
    st.markdown(text)
    st.markdown("""
                <p style='text-align: center; font-size: 12px; margin-top: 10px;'>
                网页链接：https://china.rymusic.art/
                </p>""", unsafe_allow_html=True)
    
    
    text="""
    得到的乐曲名称、流传地域与url如下：
    """
    st.markdown(text)
    input_csv = "./音乐数据/song_data_all_keywords.csv"
    music_df=pd.read_csv(input_csv,encoding='gbk')
    with st.form("myform"):
        prv = music_df.iloc[:, 1].unique()
        selected_element = st.selectbox('选择一个省份', prv)
        st.form_submit_button('提交')
        
        Series = music_df.loc[music_df['省份']==selected_element]
        st.subheader(f'共有{Series.shape[0]}条记录')
        st.dataframe(Series)

    music_df.loc[:, '音频URL'] = music_df['音频URL'].replace('', np.nan)
    music_data = music_df[music_df['音频URL'].notna()]
    text="对上述音频进行聚类，可分出三组更为接近的乐曲："
    st.markdown(text)
    cluster = Image.open('./pics/cluster_center.png')
    st.image(cluster, caption='聚类结果', use_column_width=True)
    cities_shp=gpd.read_file('./shp/cityPolygon.shp')
    text="为直观体现其地域分布，将其在地图上进行可视化。"
    st.markdown(text)
    m=mapping(cities_shp)
    cluster_csv = "./音乐数据/audio_clusters_center.csv"
    cluster_df=pd.read_csv(cluster_csv,encoding='gbk')
    cluster_df.index = cluster_df['file_name'].str.split('_').str[-1]
    cluster_df=cluster_df.drop_duplicates()
    cluster_df.to_csv('cluster_df.csv', index=False)
    point_1=[]
    point_2=[]
    point_3=[]
    for index, row in music_data.iterrows():
        city = row['城市']
        name = row['歌曲名']
        first_row = cluster_df[cluster_df.index == name].iloc[0]
        type_cluster=first_row['cluster']
        polygon_series=cities_shp[cities_shp['市']==city]
        polygon = polygon_series.iloc[0]
        area=polygon['geometry']
        bound=area.bounds
        minx = bound[0]
        miny = bound[1]
        maxx = bound[2]
        maxy = bound[3]
        while True:
            x = random.uniform(minx, maxx)
            y = random.uniform(miny, maxy)
            point = Point(x, y)
            if area.contains(point):
                if type_cluster==0:
                    point_1.append(point)   
                elif type_cluster==1:
                    point_2.append(point)
                else:
                    point_3.append(point)
                break

    for point in point_1:
        folium.Circle(
            location=[point.y, point.x],  # 使用点的坐标
            radius=10000,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=0.6
        ).add_to(m)
        
    for point in point_2:
        folium.Circle(
            location=[point.y, point.x],  # 使用点的坐标
            radius=10000,
            color='green',
            fill=True,
            fill_color='green',
            fill_opacity=0.6
        ).add_to(m)
    
    for point in point_3:
        folium.Circle(
            location=[point.y, point.x],  # 使用点的坐标
            radius=10000,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6
        ).add_to(m)
    rect_layer = folium.FeatureGroup(name='外接矩形')
    rect1 = get_minimum_bounding_rect(point_2)
    if rect1:
        ellipse_coords = [(y, x) for x, y in rect1.exterior.coords]
        folium.Polygon(
            locations=ellipse_coords,
            color='green',
            weight=2,
            fill_opacity=0
        ).add_to(rect_layer)
    rect2 = get_minimum_bounding_rect(point_3)
    if rect2:
        ellipse_coords = [(y, x) for x, y in rect2.exterior.coords]
        folium.Polygon(
            locations=ellipse_coords,
            color='red',
            weight=2,
            fill_opacity=0
        ).add_to(rect_layer)
    rect_layer.add_to(m)
    folium.LayerControl().add_to(m)
    st_folium(m, width=700, height=500)
    text="""
    从上图中我们可以看出，《茉莉花》的分异主要分为南北两派，南派从南京、扬州沿水路，分别沿长江与京杭大运河传播呈现南北走向；北派则更多地呈现东西走向，集中在中原地区。两者相比之下，南派《茉莉花》的流传范围更广，版本更多，因此更为人所熟知。
    """
    st.markdown(text)
