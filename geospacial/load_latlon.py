import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def filter_reliable_potholes(df,hole_threshold, eps, cluster_samples, reports, positive_class_ratio, reliable_col='reliable'):
    """
    Filters the input DataFrame to include only rows that are part of reliable pothole clusters.

    A reliable pothole cluster is defined as a spatial cluster identified by DBSCAN that has
    at least 15 pothole reports and at least 30 total reports.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with columns 'lat', 'lon', and 'pothole'. Additional columns are preserved.

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with the same columns as the input, containing only the reliable rows.
    """
    # Create a copy to avoid modifying the input DataFrame
    df_copy = df.copy()
    df_copy['pothole'] = np.where(df['class'] > hole_threshold, 1, 0)

    # Scale the latitude and longitude for DBSCAN
    scaler = StandardScaler()
    df_copy[['lat_scaled', 'lon_scaled']] = scaler.fit_transform(df_copy[['lat', 'lon']])

    # Apply DBSCAN to identify spatial clusters
    coords = df_copy[['lat_scaled', 'lon_scaled']].values
    dbscan = DBSCAN(eps=eps, min_samples=cluster_samples, metric='euclidean')
    df_copy['cluster'] = dbscan.fit_predict(coords)

    # Mark points that are not noise (cluster != -1) as spatially reliable
    df_copy['reliable_spatial'] = df_copy['cluster'] != -1

    # Calculate statistics for each cluster
    cluster_stats = df_copy.groupby('cluster').agg(
        total_potholes=('pothole', 'sum'),
        total_reports=('pothole', 'count')
    ).reset_index()

    # Merge cluster statistics back into the DataFrame
    df_copy = df_copy.merge(cluster_stats, on='cluster', how='left')

    # Determine if the cluster meets severity criteria
    df_copy['reliable_severity'] = (df_copy['total_potholes'] >= reports*positive_class_ratio) & (df_copy['total_reports'] >= reports)

    # Combine spatial and severity reliability into a final flag
    df_copy[reliable_col] = df_copy['reliable_spatial'] & df_copy['reliable_severity']

    # Filter to keep only reliable rows
    # reliable_df = df_copy[df_copy[reliable_col]]

    # Drop temporary columns added during processing
    temp_columns = ['pothole', 'lat_scaled', 'lon_scaled', 'cluster', 'reliable_spatial',
                    'total_potholes', 'total_reports', 'reliable_severity']
    reliable_df = df_copy.drop(columns=temp_columns, errors='ignore')

    return reliable_df
