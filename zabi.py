import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import spearmanr

# List of files with their respective skiprows values
files = {
    "R1C0Q0": ('P22406121-3_R1C0Q0_VQCOP-B004_2025-02-16-17-06-57.csv', 18),
    "R1C1Q0": ('P22406121-3_R1C1Q0_VQCOP-B004_2025-02-16-17-13-28.csv', 19),
    "R1C1Q2": ('P22406121-3_R1C1Q2_VQCOP-B004_2025-02-16-17-02-07.csv', 19),
    "R1C2Q0": ('P22406121-3_R1C2Q0_VQCOP-B004_2025-02-16-17-17-40.csv', 19),
    "R1C4Q0": ('P22406121-3_R1C4Q0_VQCOP-B004_2025-02-16-16-42-43.csv', 19),
    "R1C4Q1": ('P22406121-3_R1C4Q1_VQCOP-B004_2025-02-16-16-48-20.csv', 19),
    "R1C4Q2": ('P22406121-3_R1C4Q2_VQCOP-B004_2025-02-16-16-57-42.csv', 19),
    "R1C5Q0": ('P22406121-3_R1C5Q0_VQCOP-B004_2025-02-16-16-32-20.csv', 19),
    "R1C5Q1": ('P22406121-3_R1C5Q1_VQCOP-B004_2025-02-16-16-37-27.csv', 19),
    "R1C5Q2": ('P22406121-3_R1C5Q2_VQCOP-B004_2025-02-16-16-52-53.csv', 19),
    "R1C6Q0": ('P22406121-3_R1C6Q0_VQCOP-B004_2025-02-16-16-20-04.csv', 19),
    "R1C6Q1": ('P22406121-3_R1C6Q1_VQCOP-B004_2025-02-16-16-25-37.csv', 19),
    "R1C7Q0": ('P22406121-3_R1C7Q0_VQCOP-B004_2025-02-16-16-15-25.csv', 19),
    "R1C8Q0": ('P22406121-3_R1C8Q0_VQCOP-B004_2025-02-16-16-00-16.csv', 19),
    "R1C8Q1": ('P22406121-3_R1C8Q1_VQCOP-B004_2025-02-16-16-04-57.csv', 19)
}

# Selected radiuses based on anomalies in the report
response_variables = [
    'Dark current-r10.0', 'Dark current-r20.0', 'Dark current-r30.0', 
    'Dark current-r40.0', 'Dark current-r50.0', 'Dark current-r80.0', 
    'Dark current-r100.0', 'Dark current-r150.0', 'Dark current-r200.0', 
    'Dark current-r250.0'
]

# Step 1: Preprocess data and extract features
def preprocess_data(files):
    data = []
    raw_data = {}
    for label, (filename, skiprows) in files.items():
        df = pd.read_csv(filename, skiprows=skiprows)
        df = df.dropna()
        raw_data[label] = df
        
        for response in response_variables:
            if response in df.columns:
                mean_current = np.mean(df[response])
                std_current = np.std(df[response])
                correlation, _ = spearmanr(df['Voltage step:'], df[response])
                diffs = np.abs(np.diff(df[response]))
                max_jump = np.max(diffs) if len(diffs) > 0 else 0
                data.append([label, response, mean_current, std_current, correlation, max_jump])
                
    return pd.DataFrame(data, columns=['Device', 'Radius', 'Mean', 'StdDev', 'Correlation', 'MaxJump']), raw_data

# Step 2: Detect anomalies using Isolation Forest
def detect_anomalies(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Mean', 'StdDev', 'Correlation', 'MaxJump']])
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = model.fit_predict(X)
    df['Anomaly'] = df['Anomaly'].map({1: 'Good', -1: 'Weird'})
    return df

cluster = 3
# Step 3: Apply clustering (KMeans) to anomalous data
def cluster_anomalous_data(df):
    anomalous_data = df[df['Anomaly'] == 'Weird']
    kmeans = KMeans(n_clusters=5, random_state=42)
    anomalous_data['Cluster'] = kmeans.fit_predict(anomalous_data[['Mean', 'StdDev', 'Correlation', 'MaxJump']])
    return anomalous_data

# Step 4: Plot all clusters individually
import matplotlib.pyplot as plt

def plot_clusters_individually(df, raw_data):
    """
    Plot each cluster individually, showing the data points for each cluster in separate plots.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the anomaly and cluster information.
    raw_data (dict): A dictionary of raw data for each device.
    """
    # Filter out anomalous data
    weird_cases = df[df['Anomaly'] == 'Weird']
    
    # Iterate through each cluster and plot it separately
    for cluster in range(5):  # We have 3 clusters
        cluster_data = weird_cases[weird_cases['Cluster'] == cluster]
        
        plt.figure(figsize=(10, 6))
        
        for _, row in cluster_data.iterrows():
            device = row['Device']
            radius = row['Radius']
            
            # Plot the data for each device and radius
            plt.scatter(raw_data[device]['Voltage step:'], raw_data[device][radius], 
                        label=f'{device} - {radius}')
        
        plt.xlabel('Voltage Step')
        plt.ylabel('Dark Current')
        plt.title(f'Cluster {cluster} - Anomalous Data')
        plt.legend()
        plt.show()

# Usage: After anomaly detection and clustering
# plot_clusters_individually(anomalous_data, raw_data)


# Step 5: Save the anomalous data to Excel
def save_to_excel(data, filename):
    data.to_excel(filename, index=False)

# Running the full pipeline
data_df, raw_data = preprocess_data(files)
data_df = detect_anomalies(data_df)
anomalous_data = cluster_anomalous_data(data_df)

# Plotting the results
plot_clusters_individually(anomalous_data, raw_data)

# Save the anomalous data with clusters to Excel
save_to_excel(anomalous_data[['Device', 'Radius', 'Mean', 'StdDev', 'Correlation', 'MaxJump', 'Cluster']], 'output_file.xlsx')
