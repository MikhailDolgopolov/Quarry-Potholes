from evaluate.draw_functions import plot_feature_ranges
from exploration.data_read import load_engineered_data, load_plain_data

with open('exploration/features/features_U_anomalies.txt') as f:
    l = f.readlines()[0].rstrip('\n')
    anomaly_features = list(feat.strip('"') for feat in l.split(','))

with open('exploration/features/features_U_pressure.txt') as f:
    l = f.readlines()[0].rstrip('\n')
    pressure_features = list(feat.strip('"') for feat in l.split(','))

if __name__ == '__main__':
    # df = load_engineered_data('data/engineered/30peaks_eps5/rolled7')
    # df = load_engineered_data('data/engineered/combined/rolled5')
    # df = load_engineered_data('data/engineered/combined/rolled7')
    df = load_plain_data('data/anomalies/threshold1.3_2.0_rolled10')
    # print(df.columns)
    target = 'pothole'
    plot_feature_ranges(df[[target, *anomaly_features]])

