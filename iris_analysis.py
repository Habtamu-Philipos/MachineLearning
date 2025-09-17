import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import os

def load_iris_data():
    """Load the Iris dataset and convert it to a pandas DataFrame."""
    try:
        iris = load_iris()
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def explore_dataset(df):
    """Explore the dataset structure and check for missing values."""
    print("\n=== Dataset Exploration ===")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    return df

def clean_dataset(df):
    """Handle missing values by dropping them (if any)."""
    if df.isnull().sum().sum() > 0:
        print("\nDropping missing values...")
        df = df.dropna()
        print("Missing values after cleaning:")
        print(df.isnull().sum())
    else:
        print("\nNo missing values found.")
    return df

def analyze_data(df):
    """Perform basic statistical analysis and grouping."""
    print("\n=== Basic Data Analysis ===")
    print("\nStatistical summary of numerical columns:")
    print(df.describe())
    
    print("\nMean measurements by species:")
    group_means = df.groupby('species').mean()
    print(group_means)
    
    print("\nObservations:")
    print("- The dataset contains measurements for three Iris species.")
    print("- Each species has distinct average measurements, with virginica often having the largest values.")
    return group_means

def create_visualizations(df, group_means):
    """Create four different visualizations to explore the dataset."""
    sns.set_style("whitegrid")
    
    # 1. Line chart: Mean measurements across species
    plt.figure(figsize=(10, 6))
    for column in group_means.columns:
        plt.plot(group_means.index, group_means[column], marker='o', label=column)
    plt.title("Mean Measurements by Iris Species")
    plt.xlabel("Species")
    plt.ylabel("Measurement (cm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Fetched_Images/mean_measurements_line.png")
    plt.close()
    print("\n✓ Line chart saved to Fetched_Images/mean_measurements_line.png")
    
    # 2. Bar chart: Average sepal length per species
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='sepal length (cm)', data=df)
    plt.title("Average Sepal Length by Species")
    plt.xlabel("Species")
    plt.ylabel("Sepal Length (cm)")
    plt.tight_layout()
    plt.savefig("Fetched_Images/sepal_length_bar.png")
    plt.close()
    print("✓ Bar chart saved to Fetched_Images/sepal_length_bar.png")
    
    # 3. Histogram: Distribution of petal length
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x='petal length (cm)', bins=20, kde=True)
    plt.title("Distribution of Petal Length")
    plt.xlabel("Petal Length (cm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("Fetched_Images/petal_length_histogram.png")
    plt.close()
    print("✓ Histogram saved to Fetched_Images/petal_length_histogram.png")
    
    # 4. Scatter plot: Sepal length vs. Petal length
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
    plt.title("Sepal Length vs. Petal Length by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig("Fetched_Images/sepal_petal_scatter.png")
    plt.close()
    print("✓ Scatter plot saved to Fetched_Images/sepal_petal_scatter.png")

def main():
    """Main function to execute the analysis pipeline."""
    # Create directory for saving plots
    os.makedirs("Fetched_Images", exist_ok=True)
    
    # Load dataset
    df = load_iris_data()
    if df is None:
        return
    
    # Explore dataset
    df = explore_dataset(df)
    
    # Clean dataset
    df = clean_dataset(df)
    
    # Analyze data
    group_means = analyze_data(df)
    
    # Create visualizations
    create_visualizations(df, group_means)
    
    print("\n=== Analysis Complete ===")
    print("Key Findings:")
    print("- The Iris dataset is clean with no missing values.")
    print("- Virginica tends to have the largest measurements, followed by versicolor and setosa.")
    print("- Petal length shows clear separation between species in the scatter plot, suggesting it's a good feature for classification.")
    print("- The histogram of petal length indicates a multimodal distribution, reflecting the three species.")

if __name__ == "__main__":
    main()
