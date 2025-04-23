import pickle
import re
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import folium
from tqdm import tqdm

from exploration.data_read import read_dir_csvs, straight_predictors
from exploration.features.Separability import anomaly_features

# Constants
COLOR_PALETTE = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                 '#ff7f00', '#ffff33', '#a65628', '#f781bf']


# --------------------------
# Core Mapping Components
# --------------------------

def create_map_base(route_data: List[pd.DataFrame], route_id: int) -> folium.Map:
    """Initialize folium map with route data"""
    if not route_data:
        raise ValueError("Empty route data provided")

    if not {'lat', 'lon', 'severity'}.issubset(route_data[0].columns):
        raise ValueError("Missing required columns in route data")

    map_center = [route_data[0].lat.mean(), route_data[0].lon.mean()]
    m = folium.Map(location=map_center, zoom_start=13)

    # Add map title
    title_html = f'''
        <h3 align="center" style="font-size:20px"><b>Route {route_id}</b></h3>
        <h4 align="center" style="font-size:16px">{len(route_data)} Tracks</h4>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def add_route_tracks(m: folium.Map, tracks: List[pd.DataFrame], color: str = '#555555') -> folium.Map:
    """Add faint route tracks to the map"""
    for track in tracks:
        coords = list(zip(track['lat'], track['lon']))
        folium.PolyLine(
            coords,
            color=color,
            weight=2,
            opacity=max(0.2, 1/len(tracks))
        ).add_to(m)
    return m


# --------------------------
# Pothole Visualization
# --------------------------

def create_pothole_layer(data: pd.DataFrame,
                         layer_name: str,
                         color: str = 'blue',
                         radius_range: Tuple[float, float] = (3, 6),
                         show: bool = True) -> folium.FeatureGroup:
    """Create a configurable pothole visualization layer"""
    layer = folium.FeatureGroup(name=layer_name, show=show)

    potholes = data[data['pothole'] == 1]
    if potholes.empty:
        return layer

    min_sev = potholes['severity'].min()
    max_sev = potholes['severity'].max()
    r_min, r_max = radius_range

    for _, row in potholes.iterrows():
        severity = row['severity']
        radius = r_min + (severity - min_sev) / (max_sev - min_sev) * (r_max - r_min) if max_sev != min_sev else r_min

        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.3,
            opacity=0.7,
            tooltip=f"Severity: {severity:.2f}"
        ).add_to(layer)

    return layer


# --------------------------
# Data Handling
# --------------------------

def parse_anomaly_parameters(path: Path) -> tuple[float, ...]:
    """Extract parameters from directory name"""
    match = re.search(r"threshold([\d.]+)_([\d.]+)_rolled(\d+)", path.name)
    if not match:
        raise ValueError(f"Invalid directory format: {path.name}")
    return tuple(map(float, match.groups()[:2])) + (int(match.groups()[2]),)


def load_route_data(base_path: Path, route_id: int) -> List[pd.DataFrame]:
    """Load CSV data for a route from directory"""
    try:
        return read_dir_csvs(base_path / f'route{route_id}', pd.read_csv)
    except Exception as e:
        raise FileNotFoundError(f"Data loading failed: {str(e)}")


# --------------------------
# Main Visualization Function
# --------------------------

def visualize_anomalies(route_id: int, threshold: float = 1.3) -> Path:
    """Create interactive map comparing original and anomaly detections"""
    # Data setup
    anomalies_base = Path('data/anomalies')
    output_path = Path(f"maps/anomalies/route{route_id}_comparison.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original data from first available source
    original_data = load_route_data(Path(f'data/relabeled/threshold{threshold}'), route_id)

    if not original_data:
        raise ValueError(f"No data found for route {route_id}")
    for df in original_data:
        df['pothole'] = np.where(df['severity']>=0.3, 1, 0)
    # Initialize map
    m = create_map_base(original_data, route_id)
    m = add_route_tracks(m, original_data)

    # Add original potholes (always visible)
    m.add_child(create_pothole_layer(
        pd.concat(original_data),
        "⬤ Original Potholes",  # Black circle Unicode
        color='black',
        radius_range=(3, 5),
        show=True
    ))

    # Add anomaly layers with different parameters
    anomaly_dirs = list(anomalies_base.glob(f"threshold{threshold}*_*_rolled*"))
    for i, param_dir in enumerate(anomaly_dirs):
        try:
            color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
            params = parse_anomaly_parameters(param_dir)

            # Add colored circle to layer name
            layer_name = (
                f"<span style='color:{color}; display:inline-block; width:12px;'>⬤</span> "
                f"Anomalies: Threshold={params[0]} | Number={params[1]} | WS={params[2]}"
            )

            anomaly_data = load_route_data(param_dir, route_id)
            layer = create_pothole_layer(
                pd.concat(anomaly_data),
                layer_name,
                color=color,
                radius_range=(5, 8),
                show=(i == 0)
            )
            m.add_child(layer)

        except Exception as e:
            print(f"Skipping {param_dir.name}: {str(e)}")
            continue

    # Finalize map controls
    folium.LayerControl(collapsed=False, position='topright').add_to(m)
    m.save(str(output_path))
    return output_path


def visualize_predictions(route_id: int) -> Path:
    output_path = Path(f"maps/predictions/route{route_id}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load original data with geospatial coordinates
    original_data = load_route_data(Path('data/relabeled/extremes_w10'), route_id)
    engineered_data = load_route_data(Path('data/isoforest/2.0_rolled10'), route_id)

    # Create original pothole labels (FIX 1: Correct column assignment)
    for df in original_data:
        # Create 'pothole' COLUMN (not row) using .loc[:, 'pothole'] or direct assignment
        df['pothole'] = np.where(df['severity'] >= 0.3, 1, 0)

    # Load models
    with open('models/LVQs/glvq_hole10_[2_3]_sgd_squared-euclidean.pkl', 'rb') as f:
        lvq = pickle.load(f)

    with open('models/LogReg/Plain_54.pkl', 'rb') as f:
        logreg = pickle.load(f)

    logreg_preds = []
    for df in original_data:
        # Create a copy to preserve original data
        df_pred = df.copy()

        X_predict = df_pred[straight_predictors]  # straight_predictors should be defined

        df_pred['pothole'] = logreg.predict(X_predict.to_numpy())
        logreg_preds.append(df_pred)

    lvq_preds = []
    for df in engineered_data:
        # Create a copy to preserve original data
        df_pred = df.copy()

        X_predict = df_pred[anomaly_features]

        df_pred['pothole'] = lvq.predict(X_predict)
        lvq_preds.append(df_pred)
    m = create_map_base(original_data, route_id)
    m = add_route_tracks(m, original_data)

    # Original potholes (black)
    m.add_child(create_pothole_layer(
        pd.concat(original_data),
        "⬤ Original Potholes",
        color='red',
        radius_range=(12, 15),
        show=True
    ))

    m.add_child(create_pothole_layer(
        pd.concat(logreg_preds),
        'LogReg Potholes',
        color='red',
        radius_range=(5, 8),
    ))

    m.add_child(create_pothole_layer(
        pd.concat(lvq_preds),
        'LVQ Potholes',
        color='blue',
        radius_range=(3, 5),
    ))

    # Add layer control (FIX 5: Ensure this is added before saving)
    folium.LayerControl(collapsed=False, position='topright').add_to(m)

    m.save(str(output_path))
    return output_path
# --------------------------
# Execution
# --------------------------
bad_routes = [12, 4, 28, 9, 37]
if __name__ == '__main__':
    visualize_predictions(21)
    routes_to_process = random.sample(list(set(range(1, 39))-set(bad_routes)), k=2)

    # for rid in tqdm(routes_to_process, desc="Generating maps"):
    #     visualize_predictions(rid)
    #     except Exception as e:
    #         print(f"Failed processing route {rid}: {str(e)}")