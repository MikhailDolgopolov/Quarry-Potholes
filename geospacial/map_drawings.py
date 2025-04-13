import glob
import json
import pickle
import random
from pathlib import Path
from typing import List

import pandas as pd
import folium
from exploration.data_read import read_recoded_track, read_dir_csvs, read_raw_track


def init_map(route: List[pd.DataFrame], route_id: int = 0) -> folium.Map:
    required_columns = {'lat', 'lon', 'severity'}
    if not required_columns.issubset(route[0].columns):
        raise ValueError("Each track must have 'lat', 'lon', and 'severity' columns.")

    m = folium.Map(location=[route[0].lat.mean(), route[0].lon.mean()], zoom_start=13)

    # Add a title to the map
    title_html = f'''
        <h3 align="center" style="font-size:20px"><b>Route {route_id + 1}</b></h3>
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
                  tracks: list | pd.DataFrame,
                  name: str,
                  radius_min: float = 2,
                  radius_max: float = 6,
                  circle_color: str = 'red') -> folium.Map:
    try:
        data = pd.concat(tracks, ignore_index=True)
    except Exception:
        data = tracks

    # Filter out rows with severity <= 0
    potholes = data[data['severity'] > 0]
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
            fill_opacity=0.5,
            opacity=0.5
        ).add_to(pothole_layer)
    pothole_layer.add_to(m)
    return m


def compare_labeling(route_id: int, peaks=30) -> Path:
    originals_path = f'data/input-raw/route{route_id}'
    processed_path = f'data/preprocessed/{peaks}peaks/route{route_id}'
    originals = read_dir_csvs(originals_path, read_raw_track)
    processed = read_dir_csvs(processed_path, pd.read_csv)
    if len(originals)==0:
        raise ValueError(f"No data in route {route_id}")

    # Initialize the map using the original track list
    m = init_map(route=originals, route_id=route_id)
    m = draw_tracks(m=m, tracks=originals)

    m = draw_potholes(m=m, tracks=processed, name="Processed", circle_color='blue', radius_max=10)
    m = draw_potholes(m=m, tracks=originals, name="Original", circle_color='red', radius_max=6)

    folium.LayerControl(collapsed=False).add_to(m)

    map_path = Path(f"maps/route{route_id + 1}_pothole_map.html")
    map_path.parent.mkdir(exist_ok=True)
    m.save(str(map_path))

    return map_path


if __name__ == '__main__':
    from tqdm import tqdm
    routes = random.choices(range(1, 36), k=5)
    for route in tqdm(routes):
        compare_labeling(route)




