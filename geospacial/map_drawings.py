import glob
import json
import pickle
import random
from pathlib import Path
from typing import List

import pandas as pd
import folium
from exploration.data_read import read_dir_csvs


def init_map(route: List[pd.DataFrame], route_id: int = 0) -> folium.Map:
    required_columns = {'lat', 'lon', 'severity'}
    if not required_columns.issubset(route[0].columns):
        raise ValueError("Each track must have 'lat', 'lon', and 'severity' columns.")

    m = folium.Map(location=[route[0].lat.mean(), route[0].lon.mean()], zoom_start=13)

    # Add a title to the map
    title_html = f'''
        <h3 align="center" style="font-size:20px"><b>Route {route_id}</b></h3>
        <h4 align="center" style="font-size:16px">Tracks: {len(route)}</h4>
        '''
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def draw_tracks(m: folium.Map, tracks: List[pd.DataFrame]) -> folium.Map:
    # Draw faint polylines for each track
    for track in tracks:
        coords = list(zip(track['lat'], track['lon']))
        folium.PolyLine(coords, color='black', weight=2, opacity=0.2).add_to(m)
    return m


def draw_potholes(m: folium.Map,
                  tracks: List[pd.DataFrame] | pd.DataFrame,
                  name: str,
                  radius_min: float = 2,
                  radius_max: float = 6,
                  circle_color: str = 'red') -> folium.Map:
    if isinstance(tracks, List):
        data = pd.concat(tracks, ignore_index=True)
    elif isinstance(tracks, pd.DataFrame):
        data = tracks
    else:
        raise ValueError("tracks must be a list of DataFrames or a single DataFrame.")

    potholes = data[data['pothole'] == 1]
    if potholes.empty:
        return m

    # Compute minimum and maximum severity values
    s_min = potholes['severity'].min()
    s_max = potholes['severity'].max()

    pothole_layer = folium.FeatureGroup(name=name)
    for _, row in potholes.iterrows():
        severity = row['severity']
        if s_max != s_min:
            # Linear interpolation of severity to a radius value
            radius = radius_min + (severity - s_min) / (s_max - s_min) * (radius_max - radius_min)
        else:
            radius = radius_min
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=circle_color,
            fill=True,
            fill_opacity=0.2,
            opacity=0.7
        ).add_to(pothole_layer)
    pothole_layer.add_to(m)
    return m


def compare_labeling(route_id: int,e:int, peaks=30) -> Path:
    originals_path = f'data/renamed/route{route_id}'
    processed_path = f'data/relabeled/ws{peaks}_peaks/route{route_id}'
    cluster_path = f"data/clustered/{peaks}peaks_eps{e}/route{route_id}"
    originals = read_dir_csvs(originals_path, pd.read_csv)
    processed = read_dir_csvs(processed_path, pd.read_csv)
    clusters = read_dir_csvs(cluster_path, pd.read_csv)
    # print(len(originals), len(processed), len(clusters))
    if len(originals)==0:
        raise ValueError(f"No data in route {route_id}")

    # Initialize the map using the original track list
    m = init_map(route=originals, route_id=route_id)
    m = draw_tracks(m=m, tracks=originals)

    m = draw_potholes(m=m, tracks=processed, name="Processed", circle_color='blue', radius_max=8)
    m = draw_potholes(m=m, tracks=originals, name="Original", circle_color='red', radius_max=6)
    m = draw_potholes(m=m, tracks=clusters, name=f"Clustered e={e} meters",
                      circle_color='darkgreen', radius_min=7, radius_max=12)

    folium.LayerControl(collapsed=False).add_to(m)

    map_path = Path(f"maps/route{route_id}_clusters{e}m_pothole_map.html")
    map_path.parent.mkdir(exist_ok=True)
    m.save(str(map_path))

    return map_path


if __name__ == '__main__':
    from tqdm import tqdm
    routes = random.choices(range(1, 39), k=1)
    # for route in tqdm(routes):
    for e in tqdm([5,3]):
        compare_labeling(5, e)





