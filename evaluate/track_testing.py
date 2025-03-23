import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import folium
import pandas as pd
from tqdm import trange

from exploration.data_prep import current_transformer
from exploration.data_read import read_new_points
from helpers import discretize_to_levels, select_random_file


model_path = "HGBR_[l2_regularization0.5][learning_rate0.6][max_iter200][min_samples_leaf5][random_state42][scoringneg_mean_absolute_error][tol0.01]_top1_21.pkl"
with open(f'models/{model_path}', 'rb') as f:
    model = pickle.load(f)




def plot_lines(df):
    data = current_transformer.transform(df)
    target='class'


    Xy = data.drop(columns=['lat', 'lon'])

    X, y = Xy.drop(columns=[target]), np.clip(Xy[target], 0, 120)

    pred = model.predict(X)
    # pred = discretize_to_levels(pred, np.arange(0, 120, 30))

    plt.figure(figsize=(12, 6))

    # Plot true values (y) and predictions (pred)
    plt.plot(np.arange(len(pred)), y, label='True Values', color='blue', alpha=0.7)
    plt.plot(np.arange(len(pred)), pred, label='Predictions', color='red', linestyle='-', alpha=0.7)
    mask = (y > 0) & (pred > y)
    plt.fill_between(
        np.arange(len(pred)),
        np.where(mask, pred, y),
        # where=np.where(pred>y, True, False),
        color='black',
        alpha=0.3,
        label='Concordance'
    )

    # Add labels and title
    plt.xlabel('Seconds')
    plt.ylabel('Pothole Severity')
    plt.title('True Values vs Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Show plot
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'images/{select}.png')

def draw_map(df):
    data = current_transformer.transform(df)
    target = 'class'
    Xy = data.drop(columns=['lat', 'lon'])

    X, y = Xy.drop(columns=[target]), np.clip(Xy[target], 0, 150)

    pred = model.predict(X)
    # pred = discretize_to_levels(pred, np.arange(0, 120, 30))
    pred = np.clip(pred, 0, 130)
    results_df = pd.DataFrame({
        'lat': data['lat'],
        'lon': data['lon'],
        'true': y,
        'pred': pred,
        'deviation': pred - y  # Calculate prediction deviation
    })

    # Create base map centered on the route
    start_coord = [results_df['lat'].mean(), results_df['lon'].mean()]
    m = folium.Map(location=start_coord, zoom_start=13)

    folium.PolyLine(results_df[['lat', 'lon']], weight=0.3, alpha=0.6, color='black').add_to(m)

    # Add markers with color coding
    for idx, row in results_df.iterrows():
        if row['true'] == 0:
            continue  # Skip points with no potholes

        # Determine marker color based on prediction accuracy
        if row['deviation'] > 15:
            color = 'red'  # Overprediction
        elif row['deviation'] < -15:
            color = 'blue'  # Underprediction
        else:
            color = 'green'  # Perfect prediction

        color='red'

        # Create popup content
        popup_text = (f"True: {row['true']}<br>"
                      f"Pred: {row['pred']}<br>"
                      f"Deviation: {row['deviation']}")

        # Add marker to map
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=5 + row['true']/8,  # Size based on deviation magnitude
            color=color,
            fill=True,
            stroke=False,
            fill_opacity=1,
            popup=popup_text
        ).add_to(m)


    # Add legend
    legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 150px; height: 120px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white;
                     padding: 10px">
             <b>Legend</b><br>
             # <i style="background:red; width:20px; height:20px; display:inline-block"></i> Overprediction<br>
             # <i style="background:blue; width:20px; height:20px; display:inline-block"></i> Underprediction<br>
             # <i style="background:green; width:20px; height:20px; display:inline-block"></i> Correct<br>
             Size = Pothole Severity
         </div>
    '''
    # m.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    map_path = f'maps/{select}_pothole_map.html'
    m.save(map_path)
    # print(f"Map saved to {map_path}")

if __name__ == '__main__':
    for i in trange(1):
        try:
            ran = random.randint(1, 38)
            # ran=20
            select = f'route{ran}'
            track = read_new_points(select_random_file(f'data/routes/{select}'))
            # draw_map(track)
            plot_lines(track)
        except:
            pass