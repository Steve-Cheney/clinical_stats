import pandas as pd
import numpy as np
import time
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def read_clinical_data(path_to_file: str) -> pd.DataFrame:
    """Read in a clinical data file containing "Discoverer	Location	Diameter (mm)	Environment	Status	code_name"

        Args:
            path_to_file: String of the path to the file to read.

        Returns:
            A dataframe of the given info.
    """
    
    with open(path_to_file, 'r') as file:
        # Read first line for header
        header = file.readline().strip().split('\t')
        df = pd.read_csv(path_to_file, sep='\t', header=0, names=header)
        df['averages'] = ''
        df['std'] = ''            
    return df


def calc_vals(df: pd.DataFrame) -> pd.DataFrame:
    """Given a dataframe, calculute the means and standard deviations of the clinical files"

        Args:
            df: The clinical_data dataframe.

        Returns:
            A dataframe of the updated info.
    """

    for i, row in df.iterrows():
        code_name = row['code_name']
        path_to_file = os.path.join("inputfiles/diversityScores", f"{code_name}.diversity.txt")
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r') as file:
                values = [float(line.strip()) for line in file.readlines() if line.strip()]
                if len(values) != 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    df.at[i, 'averages'] = mean
                    df.at[i, 'std'] = std
    return df


def df_to_tsv(df: pd.DataFrame, file_path) -> None:
    """Write a DataFrame to a tab-separated (.tsv) text file.

    Args:
        df: The DataFrame to be written to the file.
        file_path: The path to the output text file.

    Returns:
        None
    """

    df.to_csv(file_path, sep='\t', index=False)


def _cluster_num(X, max):
    """Helper fuction for determining optimal cluster number"""

    distortions = []
    for i in range(1, max + 1):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Calc derivitives to find the elbow    
    differences = np.diff(distortions)
    second_differences = np.diff(differences)

    optimal_num = np.argmin(second_differences) + 1

    return optimal_num



def plot_d_scores(df: pd.DataFrame) -> None:
    """Plot the distance scores for the top 2 and lowest average diversity scores and color by KMeans clustering.

    Args:
        df: The DataFrame to be written to the file.

    Returns:
        None
    """

    df['averages'] = pd.to_numeric(df['averages'], errors='coerce')

    top_2 = df.nlargest(2, 'averages')['code_name'].tolist()
    low = df.nsmallest(1, 'averages')['code_name'].tolist()
    vals = top_2 + low

    for code_name in vals:
        path_to_file = os.path.join("inputfiles/distanceFiles", f"{code_name}.distance.txt")
        
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r') as file:
                lines = file.readlines()
            
            x_dist = []
            y_dist = []
            for line in lines:
                values = line.strip().split(',')
                x = float(values[0])
                y = float(values[1])
                x_dist.append(x)
                y_dist.append(y)

            #KMeans clustering
            X = np.array(list(zip(x_dist, y_dist)))
            optimal_num_clusters = _cluster_num(X, 10)
            kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
            kmeans.fit(X)
            cluster_labels = kmeans.labels_

            # Plot the values and color by Kmeans cluster
            sns.scatterplot(x=x_dist, y=y_dist, hue=cluster_labels, palette="crest", legend=False)
            plt.grid(True)
            plt.title(f"{code_name} Distance Plot")
            plt.savefig(f"{code_name}.png")
            plt.close()
            print(f"{code_name}.png saved to {os.getcwd()}")
            


if __name__ == "__main__":
    start_time = time.perf_counter()
    print("Reading in data")
    df = read_clinical_data("inputfiles/clinical_data.txt")
    print("Calculating stats")
    df = calc_vals(df)
    df_to_tsv(df, "clinical_data.stats.txt")
    print("Plotting distance values")
    plot_d_scores(df)

    end_time = time.perf_counter()
    print(f"Process completed in {round(end_time-start_time, 3)} seconds.")
