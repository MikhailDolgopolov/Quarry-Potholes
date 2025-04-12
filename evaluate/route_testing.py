import glob
import json
import pickle
import random
from pathlib import Path
import pandas as pd
import folium
from helpers import predict_with_my_model  # Ensure can_predict is imported
from exploration.data_read import read_truck_data
from exploration.data_feature_engineering import data_transformers

# Constants
MODEL_PATH = Path("models/LVQs/glvq_hole5_[2_2]_resampled1.1.pkl")  # Fixed path formatting
FEATURE_CONFIG = "exploration/features/features_selected.json"


def load_model(model_path: Path):
    """Load a trained model with validation"""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    if not hasattr(model, 'predict'):
        raise ValueError("Loaded object is not a valid model")

    return model


def draw_map(gps_data: pd.DataFrame, y_true: pd.Series, y_pred: pd.Series,
             r_id: int, support: int) -> Path:
    """Create a folium map comparing true and predicted potholes"""
    # Create results dataframe with proper validation
    required_columns = {'lat', 'lon'}
    if not required_columns.issubset(gps_data.columns):
        raise ValueError("GPS data missing required columns")

    results_df = pd.DataFrame({
        'lat': gps_data['lat'],
        'lon': gps_data['lon'],
        'true': y_true,
        'pred': y_pred,
    })

    # Create base map with proper centering
    start_coord = [results_df['lat'].median(), results_df['lon'].median()]
    m = folium.Map(location=start_coord, zoom_start=13, tiles='CartoDB dark_matter')

    # Add title with improved formatting
    title_html = f'''
        <h3 align="center" style="font-size:20px"><b>Route {r_id + 1}</b></h3>
        <h4 align="center" style="font-size:16px">Tracks: {support}</h4>
    '''
    m.get_root().html.add_child(folium.Element(title_html))

    # Create layered markers with improved styling
    marker_groups = {
        'True Potholes': {'color': 'red', 'data': results_df[results_df['true'] > 0]},
        'Predicted Potholes': {'color': 'blue', 'data': results_df[results_df['pred'] > 0]}
    }

    for name, config in marker_groups.items():
        fg = folium.FeatureGroup(name=name)
        for _, row in config['data'].iterrows():
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=7,
                color=config['color'],
                fill=True,
                fill_opacity=0.6,
                popup=f"{name.split()[0]}: {row['true'] if 'True' in name else row['pred']}"
            ).add_to(fg)
        fg.add_to(m)

    # Add layer control and save
    folium.LayerControl(collapsed=False).add_to(m)
    map_path = Path(f"maps/route{r_id + 1}_LVQ_map.html")
    map_path.parent.mkdir(exist_ok=True)
    m.save(str(map_path))

    return map_path



if __name__ == '__main__':
    # Data loading with error handling
    try:
        files = [glob.glob(f'data/routes/route{i}/*w.csv') for i in range(1, 36)]
        plenty = [f for f in files if len(f) > 5]
        route_id = random.choice(range(len(plenty)))
        tracks = [read_truck_data(f) for f in plenty[route_id]]
    except (FileNotFoundError, IndexError) as e:
        print(f"Data loading error: {e}")
        pass

    # Data preparation
    ws = 10
    route_data = pd.concat(tracks).dropna()
    prepared_data = data_transformers[ws].transform(route_data)

    # Feature handling with validation
    with open(FEATURE_CONFIG) as f:
        feature_sets = list(json.load(f)[f'ws_{ws}'].values())

    if not feature_sets:
        raise ValueError("No feature sets found in configuration")
    features = feature_sets[0]  # Assuming correct feature set index
    target = 'pothole'

    # Generate predictions
    X_route = prepared_data[features]
    y_route = prepared_data[target]
    predictions = predict_with_my_model(MODEL_PATH, X_route)

    # # Visualization
    # map_path = draw_map(
    #     gps_data=prepared_data[['lat', 'lon']],
    #     y_true=y_route,
    #     y_pred=pd.Series(predictions),
    #     r_id=route_id,
    #     support=len(tracks)
    # )
    #
    # print(f"Visualization saved to {map_path}")