# statreport.py

This Python script performs statistical analysis and visualization of clinical data. It reads clinical data from a text file, calculates stats such as means and standard deviations, and generates plots based on distance scores.

## Functions:

### `read_clinical_data(path_to_file: str) -> pd.DataFrame:`
Reads clinical data from a text file containing "Discoverer Location Diameter (mm) Environment Status code_name".

- **Args:**
  - `path_to_file`: String representing the path to the file to read.
- **Returns:**
  - A DataFrame containing the clinical data.

### `calc_vals(df: pd.DataFrame) -> pd.DataFrame:`
Calculates the means and standard deviations of the clinical data.

- **Args:**
  - `df`: DataFrame containing the clinical data.
- **Returns:**
  - A DataFrame with updated information including means and standard deviations.

### `df_to_tsv(df: pd.DataFrame, file_path) -> None:`
Writes a DataFrame to a tab-separated (.tsv) text file.

- **Args:**
  - `df`: DataFrame to be written to the file.
  - `file_path`: Path to the output text file.
- **Returns:**
  - None

### `plot_d_scores(df: pd.DataFrame) -> None:`
Plots the distance scores for the top 2 and lowest average diversity scores and colors by KMeans clustering.

- **Args:**
  - `df`: DataFrame containing the clinical data.
- **Returns:**
  - None

## Requirements:

```pandas
numpy
seaborn
matplotlib.pyplot
sklearn.cluster
```

## Usage:

1. Ensure you have necessary libraries installed.
2. Place your clinical data file (`clinical_data.txt`) in the `inputfiles` directory.
3. Ensure you have the appropriate files for each subject within the `inputfiles/distanceFiles` and `inputfiles/diversityScores` directories.
4. Execute the script `statreport.py`.
5. The script will generate statistical analysis and visualization results in the `clinical_data.stats.txt` file and plots for distance scores in the current directory.

Example usage:

```python3 statreport.py```
