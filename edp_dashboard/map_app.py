# coding: utf-8
# Import required libraries
import io
import sys
import os
import pickle
import copy
import pathlib
import dash
import math
import datetime as dt
import pandas as pd
from dash.dependencies import Input, Output, State, ClientsideFunction
import dash_core_components as dcc
import dash_html_components as html
import scipy.stats as scp
import plotly.graph_objects as go
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from sklearn.neighbors import KernelDensity
import geopandas as gpd
import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from iteration_utilities import deepflatten

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# ░██████╗░  ██╗░░░░░  ░█████╗░  ██████╗░  ░█████╗░  ██╗░░░░░  ░██████╗  
# ██╔════╝░  ██║░░░░░  ██╔══██╗  ██╔══██╗  ██╔══██╗  ██║░░░░░  ██╔════╝  
# ██║░░██╗░  ██║░░░░░  ██║░░██║  ██████╦╝  ███████║  ██║░░░░░  ╚█████╗░  
# ██║░░╚██╗  ██║░░░░░  ██║░░██║  ██╔══██╗  ██╔══██║  ██║░░░░░  ░╚═══██╗  
# ╚██████╔╝  ███████╗  ╚█████╔╝  ██████╦╝  ██║░░██║  ███████╗  ██████╔╝  
# ░╚════╝░  ╚══════╝   ╚════╝░  ╚═════╝░  ╚═╝░░╚═╝  ╚══════╝  ╚═════╝░░  

# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("data").resolve()

data_dir = str(DATA_PATH)
mat_file = os.path.join(data_dir, 'matricules.xlsx')
points_file = os.path.join(data_dir, 'PuntosRecarga.xlsx')
provincia_file = os.path.join(data_dir, 'provincia_listado.xlsx')
impagos_file = os.path.join(data_dir, 'impagos.xlsx')

geo_json_spain_dir = os.path.join(data_dir, 'GeoJson_SPAIN', 'Spain') 

# test_json = os.path.join(geo_json_spain_dir, 'Asturias.geojson') 


color_list = [ # list of color for autonomia
    "#005cf1",
    "#0296f1",
    "#18cdf1",
    "#067ae7",
    "#008be7",
    "#07a211",
    "#0f933b",
    "#40a649",
    "#9eaf06",
    "#6769ff",
    "#008a00",
    "#868686",
    "#ad8062",
    "#ad953b",
    "#4490b9",
    "#ffaa7f",
    "#aa5500",
    "#53563f"
]
list_autonomia_geojson = [
    "Alava.geojson",
    "Albacete.geojson",
    "Alicante.geojson",
    "Almeria.geojson",
    "Andalucia.geojson",
    "Aragon.geojson",
    "Asturias.geojson",
    "Avila.geojson",
    "Badajoz.geojson",
    "Baleares.geojson",
    "Barcelona.geojson",
    "Biskaia.geojson",
    "Burgos.geojson",
    "Caceres.geojson",
    "Cadix.geojson",
    "Cantabria.geojson",
    "Castellon.geojson",
    "Castilla-La-Mancha.geojson",
    "Catalonia.geojson",
    "Ceuta.geojson",
    "Ciudad_Real.geojson",
    "Cordoba.geojson",
    "Coruna.geojson",
    "Cuenca.geojson",
    "Extremadura.geojson",
    "Fuerteventura.geojson",
    "Galicia.geojson",
    "Girona.geojson",
    # "Gran_Canaria.geojson",
    "Granada.geojson",
    "Guadalajara.geojson",
    "Guipuzcoa.geojson",
    "Huelva.geojson",
    "Huesca.geojson",
    "Jaen.geojson",
    "La_Rioja.geojson",
    "Lanzarote.geojson",
    "Las_Palmas.geojson",
    "Leon.geojson",
    "Lleida.geojson",
    "Lugo.geojson",
    "Madrid.geojson",
    "Malaga.geojson",
    "Melilla.geojson",
    "Murcia.geojson",
    "Navarre.geojson",
    "Ourense.geojson",
    "Pais_Vasco.geojson",
    "Palencia.geojson",
    "Pontevedra.geojson",
    "Salamanca.geojson",
    "Santa_Cruz_Tenerife.geojson",
    "Segovia.geojson",
    "Sevilla.geojson",
    "Soria.geojson",
    "Tarragona.geojson",
    "Teruel.geojson",
    "Toledo.geojson",
    "Valence.geojson",
    "Valladolid.geojson",
    "Vizcaya.geojson",
    "Zamora.geojson",
    "Zaragoza.geojson"
    # canary-islands.json
    # spain-provinces.geojson
    # spain.geojson
]
# ██╗░░░░░  ░█████╗░  ░█████╗░  ██████╗░   ███████╗  ██╗  ██╗░░░░░  ███████╗  
# ██║░░░░░  ██╔══██╗  ██╔══██╗  ██╔══██╗   ██╔════╝  ██║  ██║░░░░░  ██╔════╝  
# ██║░░░░░  ██║░░██║  ███████║  ██║░░██║   █████╗░░  ██║  ██║░░░░░  █████╗░░  
# ██║░░░░░  ██║░░██║  ██╔══██║  ██║░░██║   ██╔══╝░░  ██║  ██║░░░░░  ██╔══╝░░  
# ███████╗  ╚█████╔╝  ██║░░██║  ██████╔╝   ██║░░░░░  ██║  ███████╗  ███████╗  
# ╚══════╝   ╚════╝░  ╚═╝░░╚═╝  ╚═════╝░   ╚═╝░░░░░  ╚═╝  ╚══════╝  ╚══════╝  
#matricules.xlxs
mat_cols = ['Categoria_veh_elect', 'Año', 'Mercado', 'Marca', 'Modelo', 'Provincia', 'Canal']
df_matricules = pd.read_excel(mat_file,
                              usecols=mat_cols)
#provincia_listado.xlsx
prov_cols = ['Autonomia', 'Superficie', 'PROVINCIA']
df_provincia = pd.read_excel(provincia_file,
                             usecols=prov_cols)
#impagos.xlsx
impagos_cols = ['CONSUMO_ELECTRICO_ANUAL', 'PROVINCIA', 'TIPO_CLIENTE']
df_impagos = pd.read_excel(impagos_file,
                           usecols=impagos_cols)
#PuntosRecarga.xlsx
df_points = pd.read_excel(points_file)


# ███╗░░░███╗  ███████╗  ██████╗░  ░██████╗░  ███████╗  
# ████╗░████║  ██╔════╝  ██╔══██╗  ██╔════╝░  ██╔════╝  
# ██╔████╔██║  █████╗░░  ██████╔╝  ██║░░██╗░  █████╗░░  
# ██║╚██╔╝██║  ██╔══╝░░  ██╔══██╗  ██║░░╚██╗  ██╔══╝░░  
# ██║░╚═╝░██║  ███████╗  ██║░░██║  ╚██████╔╝  ███████╗  
# ╚═╝░░░░░╚═╝  ╚══════╝  ╚═╝░░╚═╝  ░╚════╝░  ╚══════╝  

def match_region_name(reg_src, list_reg):
    """
    Matches composite names which appears in a different order
    e.g. 'Principade de Asturias' and 'Asturias Principado de'
    """
    l_src = reg_src.split()
    if len(l_src) == 1:
        return reg_src
    l = list(map(str.split, list_reg))
    l = [i for i in l if len(i) > 1]
    return ' '.join(next((s for s in l if set(l_src) == set(s)), reg_src.split()))


def replace_whole_str(sub, l):
  return next((s for s in l if sub in s), sub)

# Normalisation des regions du fichier points
df_points['region'] = df_points['region'].str.lower()
df_points['region'] = df_points['region'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

# Normalisation des noms de provinces du fichier provincia_listado
df_provincia['PROVINCIA'] = df_provincia['PROVINCIA'].str.lower()
df_provincia['PROVINCIA'] = df_provincia['PROVINCIA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_provincia['PROVINCIA'] = df_provincia['PROVINCIA'].str.replace('\(|\)', '', regex=True)

df_provincia['Autonomia'] = df_provincia['Autonomia'].str.lower()
df_provincia['Autonomia'] = df_provincia['Autonomia'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

df_provincia['Autonomia'] = df_provincia['Autonomia'].str.replace('\(|\)', '', regex=True)
df_provincia['Autonomia'] = df_provincia['Autonomia'].apply(lambda x : match_region_name(x, df_points['region'].unique()))

# Impagos normalisation
df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].str.lower()
df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
#df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].apply(lambda x : x.replace(' / ', '/'))
df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].str.replace(' / ', '/')
df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].apply(lambda x : replace_whole_str(x, df_provincia['PROVINCIA'].unique()))
df_impagos['PROVINCIA'] = df_impagos['PROVINCIA'].apply(lambda x : match_region_name(x, df_provincia['PROVINCIA'].unique()))

# Normalisation des noms des provinces du fichier matricules
df_matricules['Provincia'] = df_matricules['Provincia'].apply(lambda x : x.strip().lower())
df_matricules['Provincia'] = df_matricules['Provincia'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_matricules['Provincia'] = df_matricules['Provincia'].str.replace('\(|\)', '', regex=True)
df_matricules['Provincia'] = df_matricules['Provincia'].apply(lambda x : replace_whole_str(x, df_provincia['PROVINCIA'].unique()))

#o Join matricules w/ provincia to add Autonomia
df_merged_w_provincia = df_matricules.merge(df_provincia,
                                left_on='Provincia',
                                right_on='PROVINCIA')

# Join points w/ matricules
df_merged_w_points = df_merged_w_provincia.merge(df_points,
                                                 how='left',
                                                 left_on='Autonomia',
                                                 right_on='region')

# ███████╗  ██╗░░██╗  ██████╗░  ██╗░░░░░  ░█████╗░  ██████╗░  ░█████╗░  ████████╗  ░█████╗░  ██████╗░  ██╗░░░██╗  
# ██╔════╝  ╚██╗██╔╝  ██╔══██╗  ██║░░░░░  ██╔══██╗  ██╔══██╗  ██╔══██╗  ╚══██╔══╝  ██╔══██╗  ██╔══██╗  ╚██╗░██╔╝  
# █████╗░░  ░╚███╔╝░  ██████╔╝  ██║░░░░░  ██║░░██║  ██████╔╝  ███████║  ░░░██║░░░  ██║░░██║  ██████╔╝  ░╚████╔╝░  
# ██╔══╝░░  ░██╔██╗░  ██╔═══╝░  ██║░░░░░  ██║░░██║  ██╔══██╗  ██╔══██║  ░░░██║░░░  ██║░░██║  ██╔══██╗  ░░╚██╔╝░░  
# ███████╗  ██╔╝╚██╗  ██║░░░░░  ███████╗  ╚█████╔╝  ██║░░██║  ██║░░██║  ░░░██║░░░  ╚█████╔╝  ██║░░██║  ░░░██║░░░  
# ╚══════╝  ╚═╝░░╚═╝  ╚═╝░░░░░  ╚══════╝   ╚════╝░  ╚═╝░░╚═╝  ╚═╝░░╚═╝  ░░╚═╝░░░   ╚════╝░  ╚═╝░░╚═╝  ░░░╚═╝░░░  

# Nombre de bornes par region
df_point_per_region = df_points['region'].value_counts().rename_axis('region').reset_index(name='count_point')

# Nb matricule par region
df_veh_per_region = df_merged_w_provincia.groupby('Autonomia').agg({'Categoria_veh_elect': 'count'}).reset_index()

#Par region, nombre de bornes et voiture (Left Join)
df_merged = df_provincia.merge(df_point_per_region, how='left', left_on='Autonomia', right_on='region')\
                        .merge(df_veh_per_region, how='left', on='Autonomia')
df_merged = df_merged[['Autonomia', 'count_point', 'Categoria_veh_elect']].drop_duplicates().fillna(0)
df_merged.rename(columns={'Categoria_veh_elect': 'count_veh'}, inplace=True)

x = df_merged['Autonomia'].values.tolist()

"""
Exploring distances between points
1 - Looking the distribution of distance between points
"""

# Plot distances between pooint to see if it's something relevant 
df_points_dist = df_merged_w_points[["Autonomia","zona","latitud", "longitud"]]
df_points_dist = df_points_dist.dropna().drop_duplicates()

import geopy.distance
distance_list = []
for index_i, row_i in df_points_dist.iterrows() :
    list_dist = [geopy.distance.vincenty((row_i["latitud"], row_i["longitud"]), (row_j["latitud"], row_j["longitud"])).km 
                 for index_j, row_j in df_points_dist.iterrows() if index_i != index_j 
                 ]
    # list_dist = [geopy.distance.distance((row_i["latitud"], row_i["longitud"]), (row_j["latitud"], row_j["longitud"])).km 
    #              for index_j, row_j in df_points_dist.iterrows() if index_i != index_j 
    #              ]
    distance_list.append(min(list_dist)) 
df_points_dist["distances"] = distance_list

df_points_dist_per_region = df_points_dist['Autonomia'].value_counts().rename_axis('Autonomia').reset_index(name='count_point')
#df_points_dist_per_region["mean"] = [df_points_dist[df_points_dist["Autonomia"] == row_i["Autonomia"]].distances.mean for index_i, row_i in df_points_dist_per_region.iterrows()]
df_points_dist_per_region["mean"] = [np.mean(df_points_dist[df_points_dist["Autonomia"] == row_i["Autonomia"]].distances.tolist()) 
                                      for index_i, row_i in df_points_dist_per_region.iterrows()
                                    ]
df_points_dist_per_region["sd_deviation"] = [scp.sem(df_points_dist[df_points_dist["Autonomia"] == row_i["Autonomia"]].distances.tolist()) 
                                      for index_i, row_i in df_points_dist_per_region.iterrows()
                                    ]
# list of Autonomia for points 
df_autonomia_iter = df_points_dist["Autonomia"].drop_duplicates().to_list()

# Autonomia coords     - name - latitude - longitude
autonomia_name = [
    ["Andalucía","37.516262","-4.721374"],
    ["Aragón","41.597628","-0.905662"],
    ["Principado de Asturias","43.361395","-5.859327"],
    ["País Vasco","42.989625","-2.618927"],
    ["Cantabria","43.18284","-3.987843"],
    ["Castilla-La Mancha","39.279561","-3.097702"],
    ["Castilla y León","41.835682","-4.397636"],
    ["Cataluña","41.591159","1.520862"],
    ["Extremadura","39.493739","-6.067919"],
    ["Galicia","42.575055","-8.133856"],
    ["Madrid","40.416775","-3.70379"],
    ["Región de Murcia","38.139814","-1.366216"],
    ["Comunidad Foral de Navarra","42.695391","-1.676069"],
    ["La Rioja","42.287073","-2.539603"],
    ["Comunidad Valenciana","39.484011","-0.753281"]
]

# ██████╗░  ███████╗  ░██████╗░  ██╗  ░█████╗░  ███╗░░██╗   ░░░░░██╗  ░██████╗  ░█████╗░  ███╗░░██╗  
# ██╔══██╗  ██╔════╝  ██╔════╝░  ██║  ██╔══██╗  ████╗░██║   ░░░░░██║  ██╔════╝  ██╔══██╗  ████╗░██║  
# ██████╔╝  █████╗░░  ██║░░██╗░  ██║  ██║░░██║  ██╔██╗██║   ░░░░░██║  ╚█████╗░  ██║░░██║  ██╔██╗██║  
# ██╔══██╗  ██╔══╝░░  ██║░░╚██╗  ██║  ██║░░██║  ██║╚████║   ██╗░░██║  ░╚═══██╗  ██║░░██║  ██║╚████║  
# ██║░░██║  ███████╗  ╚██████╔╝  ██║  ╚█████╔╝  ██║░╚███║   ╚█████╔╝  ██████╔╝  ╚█████╔╝  ██║░╚███║  
# ╚═╝░░╚═╝  ╚══════╝  ░╚════╝░  ╚═╝   ╚════╝░  ╚═╝░░╚══╝   ░╚════╝░  ╚═════╝░░   ╚════╝░  ╚═╝░░╚══╝  
# crs = {'init': 'epsg:4326'}
# geo_df_com = gpd.read_file(comunidad_file, crs = crs)
# # Remove Island
# # https://github.com/simp37/GeoJson_SPAIN
# geo_df_com = geo_df_com[geo_df_com["NAME_1"] != "Islas Baleares"]
# geo_df_com = geo_df_com[geo_df_com["NAME_1"] != "Ceuta y Melilla"]

# geo_df_com['coord_label'] = geo_df_com['geometry'].apply(lambda x: x.representative_point().coords[:])
# geo_df_com['coord_label'] = [coords[0] for coords in geo_df_com['coord_label']]

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# geo_test_json = gpd.read_file(test_json, crs = crs)
# print(geo_test_json)

# with open(test_json) as data_file:
#     data_test_json = json.load(data_file)
#     # print(data_test_json)
#     # df = pd.read_csv(‘KochiMobile.csv’)
#     flatten_coords = list(deepflatten(data_test_json['features'][0]['geometry']['coordinates'][0]))
#     lon = []
#     lat = []
#     for index_i in range(0, len(flatten_coords), 2):
#         lon.append(flatten_coords[index_i])
#         lat.append(flatten_coords[index_i + 1])
#     # if(polygon.contains(Point(df[‘Longitude’][i],df[‘Latitude’][i]))):
#     name = data_test_json['features'][0]['properties']['name']
#     df_test_json = pd.DataFrame({'longitude':lon, 'latitude':lat, 'Autonomia': [name] * len(lon)})
#     # print(df_test_json.head())

# newDf.to_csv(‘KochiMobile.csv’)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ██████╗░  ░█████╗░  ░██████╗  ██╗░░██╗  ██████╗░  ░█████╗░  ░█████╗░  ██████╗░  ██████╗░  
# ██╔══██╗  ██╔══██╗  ██╔════╝  ██║░░██║  ██╔══██╗  ██╔══██╗  ██╔══██╗  ██╔══██╗  ██╔══██╗  
# ██║░░██║  ███████║  ╚█████╗░  ███████║  ██████╦╝  ██║░░██║  ███████║  ██████╔╝  ██║░░██║  
# ██║░░██║  ██╔══██║  ░╚═══██╗  ██╔══██║  ██╔══██╗  ██║░░██║  ██╔══██║  ██╔══██╗  ██║░░██║  
# ██████╔╝  ██║░░██║  ██████╔╝  ██║░░██║  ██████╦╝  ╚█████╔╝  ██║░░██║  ██║░░██║  ██████╔╝  
# ╚═════╝░  ╚═╝░░╚═╝  ╚═════╝░░  ░╚═╝░░╚═╝  ╚═════╝░   ╚════╝░  ╚═╝░░╚═╝  ╚═╝░░╚═╝  ╚═════╝░  
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Multi-dropdown options
from controls import COUNTIES, WELL_STATUSES, WELL_TYPES, WELL_COLORS


app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# Create controls
county_options = [
    {"label": str(COUNTIES[county]), "value": str(county)} for county in COUNTIES
]
autonomia_options = [
    {"label": str(row_i), "value": str(row_i)}
    for  row_i in df_merged_w_points["Autonomia"].drop_duplicates()
]

well_type_options = [
    {"label": str(WELL_TYPES[well_type]), "value": str(well_type)}
    for well_type in WELL_TYPES
]

well_status_options = [
    {"label": str(WELL_STATUSES[well_status]), "value": str(well_status)}
    for well_status in WELL_STATUSES
]

# Load data
# df = pd.read_csv(DATA_PATH.joinpath("wellspublic.csv"), low_memory=False)
# df = df[:20]
# df["Date_Well_Completed"] = pd.to_datetime(df["Date_Well_Completed"])
# df = df[df["Date_Well_Completed"] > dt.datetime(1960, 1, 1)]

# trim = df[["API_WellNo", "Well_Type", "Well_Name"]]
# trim.index = trim["API_WellNo"]
# dataset = trim.to_dict(orient="index")

# points = pickle.load(open(DATA_PATH.joinpath("points.pkl"), "rb"))


# Acces to mapbox API
mapbox_access_token = "pk.eyJ1IjoiemFrNDIwIiwiYSI6ImNrYjkyeXI5bTBhNXoyeW84c3BvamM5ZHYifQ.gwW63dwlZKG6SMDbsHnDZQ"

layout = dict(
    autosize=True,
    automargin=True,
    # margin=dict(l=30, r=30, b=20, t=40),
    margin=dict(l=10, r=10, b=30, t=30),
    # hovermode="closest",
    plot_bgcolor="#F9F9F9",
    paper_bgcolor="#F9F9F9",
    legend=dict(font=dict(size=20), orientation="h"),
    # title="Satellite Overview",

    # mapbox=dict(
    #     accesstoken=mapbox_access_token,
    #     # showlegend = False,
    #     # bearing = 0,
    #     # pitch = 0,
    #     style="light",
    #     center=dict(
    #         # lon=-3.7025599,
    #         # lat=40.4165001
    #         long = df_points_dist["longitud"].mean(),
    #         lat = df_points_dist["latitud"].mean()
    #     ),
    #     zoom=5,
    # ),
)

# Create app layout
app.layout = html.Div(
    [
        # dcc.Store(id="aggregate_data"),
        # empty Div to trigger javascript file for graph resizing
        html.Div(id="output-clientside"),
        html.Div(
            [
                html.H3(
                    "EUROBREATH.IT - EDP dashboard",
                    style={ "margin-bottom": "0px"},
                ),
            ],
            id="header",
            className="row flex-display",
            style={"margin-bottom": "25px"},
        ),
        html.Div(id="df_merged_selected_auto-clientside"),
        html.Div(id="autonomia_list_selected_auto-clientside"),
        html.Div(
            [
                html.Div(
                    [
                        # dcc.Slider(
                        #     id='number_slider',
                        #     min=0,
                        #     max=250 ,
                        #     step=10,
                        #     value=20,
                        #     marks={str(i): str(i) for i in range(0, 250, 20)},
                        #     className="dcc_control",
                        # ),
                        html.Div(id='slider-output-container'),
                        html.P(
                            "Select Autonomia region",
                            className="control_label",
                        ),
                        dcc.Dropdown(
                            id="autonomia_options",
                            options=autonomia_options,
                            multi=True,
                            value=[row_i for row_i in df_merged_w_points["Autonomia"].drop_duplicates()],
                            className="dcc_control",
                        ),
                        html.Div(
                            [
                                dcc.RadioItems(
                                    id="count_selector",
                                    options=[
                                        {"label": "All ", "value": "all"},
                                        {"label": "Vehicules only ", "value": "vehicules"},
                                        {"label": "Points only ", "value": "points"},
                                    ],
                                    value="all",
                                    labelStyle={"display": "inline-block"},
                                    className="dcc_control",
                                ),
                                # dcc.Checklist(
                                #     id="lock_selector",
                                #     options=[{"label": "Lock camera", "value": "locked"}],
                                #     className="dcc_control",
                                #     value=[],
                                # ),
                            ],
                            # id="right-column",
                            className="row flex-display",
                        ),
                    ],
                    className="pretty_container four columns",
                    id="cross-filter-options",
                ),
            ],
            className="row flex-display",
        ),
        # html.Div(
        #     [
        #         html.Div(
        #             [
        #                 html.Div(
        #                     [dcc.Graph(id="density_graph")],
        #                     # id="mapGraphContainer",
        #                     className="pretty_container",
        #                 ),
        #             ],
        #             # id="left-column",
        #             className="twelve columns",
        #         ),
        #     ],
        #     # id="container",
        #     className="row flex-display",
        # ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="main_graph")],
                    # className="pretty_container",
                    id="mapGraphContainer",
                    className="pretty_container twelve columns",
                ),
            ],
            # id="container",
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="dist_graph")],
                    # className="pretty_container",
                    id="mapContainer",
                    className="pretty_container twelve columns",
                ),
            ],
            # id="container",
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="count_graph")],
                    id="countGraphContainer",
                    className="pretty_container twelve columns",
                ),
            ],
            # id="right-column",
            className="row flex-display",
        ),
        html.Div(
            [
                html.Div(
                    [dcc.Graph(id="pie_graph")],
                    id="pieGraphContainer",
                    # className="pretty_container twelve columns",
                    className="pretty_container twelve columns",
                ),
            ],
            # id="right-column",
            className="row flex-display",
        ),
    ],
    id="mainContainer",
    style={"display": "flex", "flex-direction": "column"},
)


# Radio -> multi
# @app.callback(
#     Output("autonomia_list", "value"), 
#     [Input("autonomia_options", "value")]
# )
# def generate_list_autonomia(autonomia_options):
#     return 


# @app.callback(
#     Output('slider-output-container', 'children'),
#     [Input('number_slider', 'value')])
# def update_output(number_slider):
#     return 'Number of points to plot : {}'.format(number_slider)

# Helper functions
def human_format(num):
    if num == 0:
        return "0"

    magnitude = int(math.log(num, 1000))
    mantissa = str(int(num / (1000 ** magnitude)))
    return mantissa + ["", "K", "M", "G", "T", "P"][magnitude]


def filter_dataframe(df, autonomia_options):
    dff = df[
        df["Autonomia"].isin(autonomia_options)
    ]
    return dff


# def produce_individual(api_well_num):
#     try:
#         points[api_well_num]
#     except:
#         return None, None, None, None

#     index = list(
#         range(min(points[api_well_num].keys()), max(points[api_well_num].keys()) + 1)
#     )
#     gas = []
#     oil = []
#     water = []

#     for year in index:
#         try:
#             gas.append(points[api_well_num][year]["Gas Produced, MCF"])
#         except:
#             gas.append(0)
#         try:
#             oil.append(points[api_well_num][year]["Oil Produced, bbl"])
#         except:
#             oil.append(0)
#         try:
#             water.append(points[api_well_num][year]["Water Produced, bbl"])
#         except:
#             water.append(0)

#     return index, gas, oil, water


# def produce_aggregate(selected, year_slider):

#     index = list(range(max(year_slider[0], 1985), 2016))
#     gas = []
#     oil = []
#     water = []

#     for year in index:
#         count_gas = 0
#         count_oil = 0
#         count_water = 0
#         for api_well_num in selected:
#             try:
#                 count_gas += points[api_well_num][year]["Gas Produced, MCF"]
#             except:
#                 pass
#             try:
#                 count_oil += points[api_well_num][year]["Oil Produced, bbl"]
#             except:
#                 pass
#             try:
#                 count_water += points[api_well_num][year]["Water Produced, bbl"]
#             except:
#                 pass
#         gas.append(count_gas)
#         oil.append(count_oil)
#         water.append(count_water)

#     return index, gas, oil, water


# Create callbacks
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="resize"),
    Output("output-clientside", "children"),
    [Input("count_graph", "figure")],
)


# @app.callback(
#     Output("aggregate_data", "data"),
#     [
#         Input("well_statuses", "value"),
#         Input("well_types", "value"),
#         Input("year_slider", "value"),
#     ],
# )
# def update_production_text(well_statuses, well_types, year_slider):

#     dff = filter_dataframe(df, well_statuses, well_types, year_slider)
#     selected = dff["API_WellNo"].values
#     index, gas, oil, water = produce_aggregate(selected, year_slider)
#     return [human_format(sum(gas)), human_format(sum(oil)), human_format(sum(water))]


# # Radio -> multi
# @app.callback(
#     Output("well_statuses", "value"), 
#     [
#         Input("well_status_selector", "value")
#     ]
# )
# def display_status(selector):
#     if selector == "all":
#         return list(WELL_STATUSES.keys())
#     elif selector == "active":
#         return ["AC"]
#     return []


# Radio -> multi
# @app.callback(Output("well_types", "value"), [Input("well_type_selector", "value")])
# def display_type(selector):
#     if selector == "all":
#         return list(WELL_TYPES.keys())
#     elif selector == "productive":
#         return ["GD", "GE", "GW", "IG", "IW", "OD", "OE", "OW"]
#     return []


# ███╗░░░███╗  ░█████╗░  ██████╗░  
# ████╗░████║  ██╔══██╗  ██╔══██╗  
# ██╔████╔██║  ███████║  ██████╔╝  
# ██║╚██╔╝██║  ██╔══██║  ██╔═══╝░  
# ██║░╚═╝░██║  ██║░░██║  ██║░░░░░  
# ╚═╝░░░░░╚═╝  ╚═╝░░╚═╝  ╚═╝░░░░░  
@app.callback(
    Output("main_graph", "figure"),
    [
        # Input("well_statuses", "value"),
        # Input("well_types", "value"),
        # Input("year_slider", "value"),
        Input("autonomia_options", "value"),
        Input("count_selector", "value"),
        # Input("number_slider", "value"),
    ],
    [
        # State("lock_selector", "value"),
        State("main_graph", "relayoutData")
    ],
)
def map_graph(
    # selector,
    main_graph_layout, autonomia_options, count_selector
# number_slider
    ):
    zoom = 5
    lonInitial = np.mean(df_points_dist["longitud"])
    latInitial = np.mean(df_points_dist["latitud"])
    bearing = 0
    data_map = []
    color_i = 0
    for autonomia_i in df_autonomia_iter:
        dff = df_points_dist[df_points_dist["Autonomia"] == autonomia_i][["Autonomia", "latitud", "longitud"]]

        data_map.append(
            Scattermapbox(

                lat=dff.latitud,
                lon=dff.longitud,
                mode="markers+text",
                # text = autonomia_i,
                name = autonomia_i,
                # label = "Charging points",
                # legend = True,
                hoverinfo="lat+lon+text",

                marker=dict(
                    opacity=0.8,
                    size=20,
                    # symbol = 'cross',
                    color = color_list[color_i],
                    # colorbar=dict(
                    #     title="Density of points",
                    #     x=0.93,
                    #     xpad=0,
                    #     nticks=24,
                    #     tickfont=dict(color="#d8d8d8"),
                    #     titlefont=dict(color="#d8d8d8"),
                    #     thicknessmode="pixels",
                    # ),
                ),
            )
        )
        color_i += 1

    # Plot of important locations on the map
    data_map.append(
        Scattermapbox(
            lat=[auto[1] for auto in autonomia_name],
            lon=[auto[2] for auto in autonomia_name],
            mode = "text",
            # mode="text",
            name = "Autonomia name",
            # hoverinfo="text",
            text=[auto[0] for auto in autonomia_name],
            textposition = "middle center",
            textfont=dict(
                family="sans serif",
                size=18,
                color="#000"
            )
            # marker=dict(size=20, color="#64d036"),
        ),
    )

    list_layer = []
    color_i = 0
    for file in list_autonomia_geojson:
        # filename = os.fsdecode(file)
        # print(os.path.join(geo_json_spain_dir, file))
        if file.endswith(".geojson"):
            file_to_load = os.path.join(geo_json_spain_dir, file)
            # try:
            with io.open(file_to_load,'r',encoding='latin1') as data_file:
                data_json = json.load(data_file)
                try:
                    list_layer.append(
                        {
                            'source': {
                                'type': "FeatureCollection",
                                'features': [{
                                    'type': "Feature",
                                    'opacity' : 0.1,
                                    'geometry': {
                                        'type': "MultiPolygon",
                                        'coordinates': data_json['features'][0]['geometry']['coordinates'],
                                    }
                                }]
                            },
                            'type': "fill",
                            # 'below': "traces",
                            'color': color_list[-color_i%len(color_list)],
                            'opacity' : 0.1,
                            'name' : data_json['features'][0]['properties']['name'],
                        }
                    )
                except:
                #     print("Failed")
                    pass
        color_i += 1


    return go.Figure(
        data = data_map,
        layout=Layout(
            autosize=True,
            margin=go.layout.Margin(l=10, r=10, t=20, b=10),
            showlegend=True,
            # textposition="bottom center",
            legend=dict(font=dict(size=15), orientation="h"),
            # title="Satellite Overview",
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),
                style="light",
                # style="dark",
                bearing=bearing,
                zoom=zoom,
                layers=list_layer,
            ),
            paper_bgcolor = '#f9f9f9',
            plot_bgcolor = '#f9f9f9',

            updatemenus=[
                dict(
                    buttons=(
                        [
                            dict(
                                args=[
                                    {
                                        "mapbox.zoom": 5,
                                        "mapbox.center.lon": lonInitial,
                                        "mapbox.center.lat": latInitial,
                                        "mapbox.bearing": 0,
                                        "mapbox.pitch": 0,
                                        "mapbox.style": "light",
                                    }
                                ],
                                label="Reset Zoom",
                                method="relayout",
                            )
                        ]
                    ),
                    # direction="left",
                    pad={"r": 10, "t": 20, "b": 10, "l": 10},
                    showactive=False,
                    type="buttons",
                    x=0.45,
                    y=0.02,
                    xanchor="left",
                    yanchor="bottom",
                    bgcolor="#000",
                    borderwidth=1,
                    bordercolor="#6d6d6d",
                    font=dict(color="#FFFFFF"),
                )
            ],
        ),
    )
"""
# ██████╗░  ███████╗  ███╗░░██╗  ░██████╗  ██╗  ████████╗  ██╗░░░██╗   ███╗░░░███╗  ░█████╗░  ██████╗░  
# ██╔══██╗  ██╔════╝  ████╗░██║  ██╔════╝  ██║  ╚══██╔══╝  ╚██╗░██╔╝   ████╗░████║  ██╔══██╗  ██╔══██╗  
# ██║░░██║  █████╗░░  ██╔██╗██║  ╚█████╗░  ██║  ░░░██║░░░  ░╚████╔╝░   ██╔████╔██║  ███████║  ██████╔╝  
# ██║░░██║  ██╔══╝░░  ██║╚████║  ░╚═══██╗  ██║  ░░░██║░░░  ░░╚██╔╝░░   ██║╚██╔╝██║  ██╔══██║  ██╔═══╝░  
# ██████╔╝  ███████╗  ██║░╚███║  ██████╔╝  ██║  ░░░██║░░░  ░░░██║░░░   ██║░╚═╝░██║  ██║░░██║  ██║░░░░░  
# ╚═════╝░  ╚══════╝  ╚═╝░░╚══╝  ╚═════╝░░  ╚═╝  ░░╚═╝░░░  ░░░╚═╝░░░   ╚═╝░░░░░╚═╝  ╚═╝░░╚═╝  ╚═╝░░░░░  
@app.callback(
    Output("density_graph", "figure"),
    [
        Input("autonomia_options", "value"),
        Input("count_selector", "value"),
        # Input("number_slider", "value"),
    ],
    [
        # State("lock_selector", "value"),
        State("density_graph", "relayoutData")
    ],
)
def density_graph(
    main_graph_layout, autonomia_options, count_selector
    ):
    zoom = 5
    lonInitial = np.mean(df_points_dist["longitud"])
    latInitial = np.mean(df_points_dist["latitud"])
    bearing = 0

    # Calculate the point density
    # coord_stack = np.vstack([df_points_dist["longitud"],df_points_dist["latitud"]])
    # z = gaussian_kde(coord_stack)(coord_stack)
    # cmap = LinearSegmentedColormap.from_list('custom blue', 
    #                                              [(0,    '#ffff00'),
    #                                               (0.25, '#002266'),
    #                                               (1,    '#002266')], N=df_points_dist.shape[0])
    # print("cmap = ",cmap)
    # Sort the points by density, so that the densest points are plotted last
    # idx = z.argsort()
    # x, y, z = x[idx], y[idx], z[idx]
    # fig, ax = plt.subplots()
    # ax.scatter(x, y, c=z, s=50, edgecolor='')
    # color_list = [ ]
    data_dens = []

    # data_dens.append(
    #     go.Choropleth(
    #         locations = df_test_json[['longitude', 'latitude']],
    #         # mode='markers',
    #         z = [1] * len(df_test_json['longitude']),
    #         colorscale = 'Reds',
    #         colorbar_title = "Millions USD",
    #         # name = mob
    #     )
    # )
    # data_dens.append(
    #     Scattermapbox(
    #         fill = "toself",
    #         # lon = [-74, -70, -70, -74], lat = [47, 47, 45, 45],
    #         marker = { 'size': 10, 'color': "orange" },
    #         mode='text',
    #         lat=df_test_json['latitude'].tolist(),
    #         lon=df_test_json['longitude'].tolist(),
    #         # mode='markers+lines+fill',
    #         # marker=dict(
    #         #     size=20, 
    #         #     color="#64d036"
    #         # ),
    #         # text=df_test_json["Autonomia"],
    #         # name = mob
    #     )
    # )
    list_layer = []
    color_i = 0
    for file in list_autonomia_geojson:
        # filename = os.fsdecode(file)
        print(os.path.join(geo_json_spain_dir, file))
        if file.endswith(".geojson"):
            file_to_load = os.path.join(geo_json_spain_dir, file)
            # try:
            with io.open(file_to_load,'r',encoding='latin1') as data_file:
                data_json = json.load(data_file)
                try:
                    list_layer.append(
                        {
                            'source': {
                                'type': "FeatureCollection",
                                'features': [{
                                    'type': "Feature",
                                    'opacity' : 0.1,
                                    'geometry': {
                                        'type': "MultiPolygon",
                                        'coordinates': data_json['features'][0]['geometry']['coordinates'],
                                    }
                                }]
                            },
                            'type': "fill",
                            # 'below': "traces",
                            'color': color_list[-color_i%len(color_list)],
                            'opacity' : 0.1,
                            'name' : data_json['features'][0]['properties']['name'],
                        }
                    )
                except:
                    print(data_json)
                #     print("Failed")
                    pass
        color_i += 1
    return go.Figure(
        data = data_dens,
        layout=Layout(
            autosize = True,
            hovermode = 'closest',
            margin = go.layout.Margin(l=10, r=10, t=20, b=10),
            showlegend=True,
            # textposition="bottom center",
            legend=dict(font=dict(size=15), orientation="h"),
            title="Satellite Overview",
            mapbox=dict(
                accesstoken=mapbox_access_token,
                center=dict(lat=latInitial, lon=lonInitial),
                style="light",
                # style="dark",
                bearing=bearing,
                zoom=zoom,
                layers=list_layer,
            ),
        ),
    )

"""
# ██████╗░  ░█████╗░  ██╗░░░██╗  ███╗░░██╗  ████████╗  
# ██╔══██╗  ██╔══██╗  ██║░░░██║  ████╗░██║  ╚══██╔══╝  
# ██║░░╚═╝  ██║░░██║  ██║░░░██║  ██╔██╗██║  ░░░██║░░░  
# ██║░░██╗  ██║░░██║  ██║░░░██║  ██║╚████║  ░░░██║░░░  
# ╚█████╔╝  ╚█████╔╝  ╚██████╔╝  ██║░╚███║  ░░░██║░░░  
# ░╚════╝░   ╚════╝░  ░╚═════╝░  ╚═╝░░╚══╝  ░░╚═╝░░░  
# Selectors -> count graph
@app.callback(
    Output("count_graph", "figure"),
    [
        Input("autonomia_options", "value"),
        # Input("number_slider", "value"),
        Input("count_selector", "value")
    ],
)
def make_count_figure(autonomia_options, count_selector
                      # , number_slider
                      ):
    layout_count = copy.deepcopy(layout)
    # autonomias = df_merged['Autonomia'].unique()
    # dff = filter_dataframe(df, autonomia_options)
    # g = dff[["API_WellNo", "Date_Well_Completed"]]
    # g.index = g["Date_Well_Completed"]
    # g = g.resample("A").count()
    df_merged_selected_auto = df_merged[df_merged["Autonomia"].isin(autonomia_options)]
    autonomia_list_selected_auto = df_merged_selected_auto['Autonomia'].unique()

    data = []
    if count_selector == "all" or count_selector == "points":
        data.append(
            dict(
                type="bar",
                # mode="markers",
                x=autonomia_list_selected_auto,
                y=df_merged_selected_auto['count_point'],
                name="nb_points",
                # opacity=0,
                # hoverinfo="skip",
            )
        )
    if count_selector == "all" or count_selector == "vehicules" :
        data.append(
            dict(
                type="bar",
                x=autonomia_list_selected_auto,
                y=df_merged_selected_auto['count_veh'],
                name="nb_vehicles",
                # marker=dict(color=colors),
            )
        )

    layout_count["title"] = "Number of points and vehicle per autonomia"
    layout_count["dragmode"] = "select"
    layout_count["showlegend"] = True
    layout_count["autosize"] = False
    layout_count["margin"] = go.layout.Margin(l=10, r=10, t=20, b=10)
    layout_count["paper_bgcolor"] = '#f9f9f9'
    layout_count["plot_bgcolor"] = '#f9f9f9'
    figure = dict(data=data, layout=layout_count)
    return figure


# ██████╗░  ██╗  ░██████╗  ████████╗  ░█████╗░  ███╗░░██╗  ██████╗░  ███████╗  ░██████╗  
# ██╔══██╗  ██║  ██╔════╝  ╚══██╔══╝  ██╔══██╗  ████╗░██║  ██╔══██╗  ██╔════╝  ██╔════╝  
# ██║░░██║  ██║  ╚█████╗░  ░░░██║░░░  ███████║  ██╔██╗██║  ██║░░╚═╝  █████╗░░  ╚█████╗░  
# ██║░░██║  ██║  ░╚═══██╗  ░░░██║░░░  ██╔══██║  ██║╚████║  ██║░░██╗  ██╔══╝░░  ░╚═══██╗  
# ██████╔╝  ██║  ██████╔╝  ░░░██║░░░  ██║░░██║  ██║░╚███║  ╚█████╔╝  ███████╗  ██████╔╝  
# ╚═════╝░  ╚═╝  ╚═════╝░░  ░░╚═╝░░░  ╚═╝░░╚═╝  ╚═╝░░╚══╝  ░╚════╝░  ╚══════╝  ╚═════╝░░  

# Selectors -> dist_graph graph
@app.callback(
    Output("dist_graph", "figure"),
    [
        Input("autonomia_options", "value"),
        # Input("number_slider", "value"),
        Input("count_selector", "value")
    ],
)
def make_dist_figure(autonomia_options, count_selector
                     # , number_slider
                     ):
    layout_dist = copy.deepcopy(layout)
    # autonomias = df_merged['Autonomia'].unique()
    # dff = filter_dataframe(df, autonomia_options)
    # g = dff[["API_WellNo", "Date_Well_Completed"]]
    # g.index = g["Date_Well_Completed"]
    # g = g.resample("A").count()
    df_merged_selected_auto = df_merged[df_merged["Autonomia"].isin(autonomia_options)]
    autonomia_list_selected_auto = df_merged_selected_auto['Autonomia'].unique()

    nb_autonomia=0
    for index_i, row_i in df_points_dist_per_region.iterrows():
      if row_i.count_point > 2:
        nb_autonomia+=1
    ncols = 2
    nrows = nb_autonomia//2 + nb_autonomia%2

    # plt.gcf().subplots_adjust( wspace = 0.2, hspace = 0.2)

    # axs = fig.add_subplot(nrows, ncols, i )
    # axs.hist(x = df_points_dist[df_points_dist["Autonomia"] == row_i["Autonomia"]].distances.tolist(), bins=5, rwidth=0.9)
    # axs.set_title('Nearest distances in : {} ({} points mean = {:.4f})'.format(row_i["Autonomia"], row_i["count_point"], row_i["mean"]))
    # axs.set_xlabel('distance(km)')
    # axs.set_ylabel('Points number')

    data_dist = []

    i = 1
    for index_i, row_i in df_points_dist_per_region.iterrows():
      # if row_i.count_point > 2:
        data_dist.append(
            go.Histogram(
                x=df_points_dist[df_points_dist["Autonomia"] == row_i["Autonomia"]].distances.tolist(),
                hovertemplate = "%{y} Points \n in <br> (%{x}) km<br>",
                # histfunc = 'max',
                # histnorm='percent',
                name='{} ({} points, mean = {:.4f} km)'.format(row_i["Autonomia"], row_i["count_point"], row_i["mean"]),
                xbins=dict( start=0,  end = max(df_points_dist.distances.tolist())),
                # xbins=dict( start=-4.0, end=3.0, size=0.5),
                # marker_color='#EB89B5',
                opacity=0.8
            ),
        )
        i+= 1

    return go.Figure(
        data = data_dist,
        layout = Layout(
            title_text='Distances to the closest points', # title of plot
            legend=dict(font=dict(size=15), orientation="h"),
            margin=go.layout.Margin(l=10, r=10, t=40, b=10),
            # hovermode= 'range',
            # hoverformat= '', 
            xaxis_title_text='Distance(km)', # xaxis label
            yaxis_title_text='Count', # yaxis label
            bargap=0.2,                # gap between bars of adjacent location coordinates
            bargroupgap=0.1,                # gap between bars of the same location coordinates
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9",

        )
    )

# ██████╗░  ██╗  ███████╗  
# ██╔══██╗  ██║  ██╔════╝  
# ██████╔╝  ██║  █████╗░░  
# ██╔═══╝░  ██║  ██╔══╝░░  
# ██║░░░░░  ██║  ███████╗  
# ╚═╝░░░░░  ╚═╝  ╚══════╝  
# Selectors, count graph -> pie graph
@app.callback(
    Output("pie_graph", "figure"),
    [
        Input("autonomia_options", "value"),
        # Input("number_slider", "value"),
        Input("count_selector", "value")
    ]
)
def make_pie_figure(autonomia_options, count_selector
                    # , number_slider
                    ):

    autonomia_list = df_merged['Autonomia'].unique()
    layout_pie = copy.deepcopy(layout)
    # dff = filter_dataframe(df, well_statuses, well_types, year_slider)
    # selected = dff["API_WellNo"].values
    # index, gas, oil, water = produce_aggregate(selected, year_slider)
    # aggregate = dff.groupby(["Well_Type"]).count()

    df_merged_selected_auto = df_merged[df_merged["Autonomia"].isin(autonomia_options)]
    autonomia_list_selected_auto = df_merged_selected_auto['Autonomia'].unique()

# fig = px.pie(df_merged[["Autonomia", "count_veh"]], values='count_veh', names='Autonomia', title='Distribution of cars', hole=.1)
# fig.update_traces(textposition='inside', textinfo='value+percent+label')
# fig = px.pie(df_merged[["Autonomia", "count_point"]], values='count_point', names='Autonomia', title='Distribution of Points', hole=.1)
# fig.update_traces(textposition='inside', textinfo='value+percent+label')
    data = []

    # Pie placements
    if count_selector == "all":
        domain_points = {"x": [0, 0.5], "y": [0, 1]}
        domain_cars = {"x": [0.5, 1], "y": [0, 1]}
    if count_selector == "points":
        domain_points = {"x": [0.37, 0.72], "y": [0, 1]}
        domain_cars = {"x": [0.8, 1], "y": [0, 1]}
    if count_selector == "vehicules" :
        domain_points = {"x": [0, 0.2], "y": [0, 1]}
        domain_cars = {"x": [0.37, 0.72], "y": [0, 1]}

    if count_selector == "all" or count_selector == "points":
        data.append(
            dict(
                type="pie",
                # labels=["Gas", "Oil", "Water"],
                labels=autonomia_list_selected_auto,
                # values = [sum(gas), sum(oil), sum(water)],
                values = df_merged_selected_auto['count_point'],
                name="Distribution of points",
                # text=[
                #     "Total Gas Produced (mcf)",
                #     "Total Oil Produced (bbl)",
                #     "Total Water Produced (bbl)",
                # ],
                hoverinfo="name+value+percent+label",
                textinfo="value+percent+label",
                # textinfo="label+percent+name",
                textposition='inside',
                hole=0.2,
                title="Points", 
                # marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
                domain=domain_points,
            )
        )
    if count_selector == "all" or count_selector == "vehicules" :
        data.append(
            dict(
                type="pie",
                # labels=["Gas", "Oil", "Water"],
                labels=autonomia_list_selected_auto,
                # values = [sum(gas), sum(oil), sum(water)],
                values = df_merged_selected_auto['count_veh'],
                name="Distribution of cars",
                title="Cars", 
                # text=[
                #     "Total Gas Produced (mcf)",
                #     "Total Oil Produced (bbl)",
                #     "Total Water Produced (bbl)",
                # ],
                hoverinfo="name+value+percent+label",
                textinfo="value+percent+label",
                textposition='inside',
                hole=0.2,
                # marker=dict(colors=["#fac1b7", "#a9bb95", "#92d8d8"]),
                domain=domain_cars,
                margin=go.layout.Margin(l=10, r=10, t=40, b=10),
            )
        )
    layout_pie["title"] = "Pie Chart of Points and cars in Autonomia regions"
    # layout_pie["font"] = dict(color="#777777")
    # layout_pie["legend"] = dict( font=dict( # color="#CCCCCC", size="10"), orientation="h", bgcolor="rgba(0,0,0,0)")
    layout_pie["legend"] = dict(font=dict(size=20), orientation="h")
    layout_pie["paper_bgcolor"] = '#f9f9f9'
    layout_pie["plot_bgcolor"] = '#f9f9f9',
    figure = dict(data=data, layout=layout_pie)
    return figure

# Main
if __name__ == "__main__":
    app.run_server(debug=True)
