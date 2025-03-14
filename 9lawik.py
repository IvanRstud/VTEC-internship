import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

def preprocess_data(files):
    data = []
    raw_data = {}
    for label, (filename, skiprows) in files.items():
        df = pd.read_csv(filename, skiprows=skiprows)
        df = df.dropna()
        raw_data[label] = df
        
        # Calculate statistical features for anomaly detection
        for response in response_variables:
            if response in df.columns:
                mean_current = np.mean(df[response])
                std_current = np.std(df[response])
                correlation, _ = spearmanr(df['Voltage step:'], df[response])
                
                # Identify large jumps or drops
                diffs = np.abs(np.diff(df[response]))
                max_jump = np.max(diffs) if len(diffs) > 0 else 0
                
                data.append([label, response, mean_current, std_current, correlation, max_jump])
                
    return pd.DataFrame(data, columns=['Device', 'Radius', 'Mean', 'StdDev', 'Correlation', 'MaxJump']), raw_data

def detect_anomalies(df):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[['Mean', 'StdDev', 'Correlation', 'MaxJump']])
    
    model = IsolationForest(contamination=0.1, random_state=42)
    df['Anomaly'] = model.fit_predict(X)
    df['Anomaly'] = df['Anomaly'].map({1: 'Good', -1: 'Weird'})
    
    return df

def plot_weird_cases(df, raw_data):
    weird_cases = df[df['Anomaly'] == 'Weird']
    for _, row in weird_cases.iterrows():
        device = row['Device']
        radius = row['Radius']
        plt.figure(figsize=(8, 5))
        plt.scatter(raw_data[device]['Voltage step:'], raw_data[device][radius], label=f'{device} - {radius}', color='red')
        plt.xlabel('Voltage Step')
        plt.ylabel('Dark Current')
        plt.title(f'Weird Behavior in {device} - {radius}')
        plt.legend()
        plt.show()

# Run preprocessing and anomaly detection
data_df, raw_data = preprocess_data(files)
data_df = detect_anomalies(data_df)

# Display results
print(data_df)

# Plot weird cases
# plot_weird_cases(data_df, raw_data)

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you already have the preprocessed data in data_df with 'Mean', 'StdDev', 'Correlation', 'MaxJump'
# Step 1: Anomaly detection using Isolation Forest
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
data_df['Anomaly'] = anomaly_detector.fit_predict(data_df[['Mean', 'StdDev', 'Correlation', 'MaxJump']])

# Step 2: Filter out the anomalies
anomalous_data = data_df[data_df['Anomaly'] == -1]

# Step 3: Cluster the anomalous data using K-Means
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust number of clusters based on your data
anomalous_data['Cluster'] = kmeans.fit_predict(anomalous_data[['Mean', 'StdDev', 'Correlation', 'MaxJump']])

# Step 4: Visualize the clusters of anomalies
plt.figure(figsize=(8, 6))
plt.scatter(anomalous_data['Mean'], anomalous_data['StdDev'], c=anomalous_data['Cluster'], cmap='viridis')
plt.xlabel('Mean')
plt.ylabel('StdDev')
plt.title('Clusters of Anomalous Data')
plt.colorbar(label='Cluster')
plt.show()

# Display the anomalous data with assigned clusters
print(anomalous_data[['Device', 'Radius', 'Mean', 'StdDev', 'Correlation', 'MaxJump', 'Cluster']])

def save_to_excel(data, filename):
  
    data.to_excel(filename, index=False)




save_to_excel(anomalous_data[['Device', 'Radius', 'Mean', 'StdDev', 'Correlation', 'MaxJump', 'Cluster']], 'output_file.xlsx')