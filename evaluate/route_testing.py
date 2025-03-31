import glob
import pickle
import random
from collections import Counter
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import folium
import pandas as pd
from tqdm import trange

from clustering.LVQ import features_for_LVQ
from exploration.data_prep import data_transformers
from exploration.data_read import read_new_points, read_truck_data
from helpers import discretize_to_levels, select_random_file


model_path = "models\LVQs\glvq_hole5_[2_2]_resampled1.1.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

def draw_map(df, rolling_window:int, r_id, support):
    data = data_transformers[rolling_window].transform(df)

    X, y = data[features_for_LVQ], data[target]

    pred = model.predict(X)
    print(Counter(y), Counter(pred))
    # pred = discretize_to_levels(pred, np.arange(0, 120, 30))
    results_df = pd.DataFrame({
        'lat': df['lat'],
        'lon': df['lon'],
        'true': y,
        'pred': pred,
    })

    # Create base map centered on the route
    start_coord = [results_df['lat'].mean(), results_df['lon'].mean()]
    m = folium.Map(location=start_coord, zoom_start=13)

    # Add title to the map
    title_html = '''
           <h3 align="center" style="font-size:20px"><b>Route {}</b></h3>
           <h4 align="center" style="font-size:16px">Number of tracks: {}</h4>
       '''.format(r_id+1, support)
    m.get_root().html.add_child(folium.Element(title_html))
    # Create feature groups for true and predicted potholes
    true_potholes = folium.FeatureGroup(name='True Potholes')
    predicted_potholes = folium.FeatureGroup(name='Predicted Potholes')

    # Add circles for true potholes
    for _, row in results_df[y>0].iterrows():
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=7,
            color='red',
            stroke=False,
            fill=True,
            fill_color='red',
            fill_opacity=0.6,
            popup=f"True: {row['true']}"
        ).add_to(true_potholes)

    # Add circles for predicted potholes
    for _, row in results_df[pred>0].iterrows():
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=10,
            color='blue',
            stroke=False,
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"Predicted: {row['pred']}"
        ).add_to(predicted_potholes)

    # Add feature groups to the map
    true_potholes.add_to(m)
    predicted_potholes.add_to(m)

    # Add layer control to the map
    folium.LayerControl().add_to(m)

    # Save the map
    map_path = f'maps/route{r_id+1}_LVQpothole_map.html'
    m.save(map_path)
    print(f"Map saved to {map_path}")




if __name__ == '__main__':
    target='hole'
    files = [glob.glob(f'data/routes/route{i}/*w.csv') for i in range(1, 36)]
    # pprint(files[4])
    plenty = [files[i] for i in range(len(files)) if len(files[i])>5]
    routeID = random.randint(0, len(plenty)-1)
    route = plenty[routeID]
    tracks = [read_truck_data(route[i]) for i in range(len(route))]
    # print(track.describe())
    # print(track.columns)
    # print(track['lat', 'lon'])
    route_data = pd.concat(tracks)
    draw_map(route_data, routeID, len(route))
    # for i in trange(5):
    #     try:
    #
    #     except:
    #         pass