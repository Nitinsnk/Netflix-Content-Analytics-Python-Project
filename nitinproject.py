import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load your dataset
df = pd.read_csv(r"C:\Users\DELL\Downloads\netflix_content_2023_Python_Project.csv")

# the first few rows to get familiar
print(" First 5 rows of the dataset:\n")
print(df.head())

# Number of rows and columns
print("\n Shape of the dataset:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Column names
print("\n Column names:")
print(df.columns.tolist())

# Data types of each column
print("\n Data types:")
print(df.dtypes)

# Checking for missing values
print("\n Missing values in each column:")
print(df.isnull().sum())

# Summary statistics for numeric columns
print("\n Summary statistics for numerical data:")
print(df.describe())

# Look at top values in categorical columns
print("\n Top values in each text column:")
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head())

# Check for duplicate rows
print(f"\n Number of duplicate rows: {df.duplicated().sum()}")

# Clean 'Hours Viewed'
df['Hours Viewed'] = df['Hours Viewed'].str.replace(",", "")
df['Hours Viewed'] = pd.to_numeric(df['Hours Viewed'], errors='coerce')

# Convert 'Release Date' to datetime and extract year
df['Release Date'] = pd.to_datetime(df['Release Date'], format="%d-%m-%Y", errors='coerce')
df['Release Year'] = df['Release Date'].dt.year

# Plot settings
sns.set(style="whitegrid")

# OBJECTIVE 1: Global Content Distribution by Language
plt.figure(figsize=(10, 6))
language_counts = df['Language Indicator'].value_counts().head(15).reset_index()
language_counts.columns = ['Language', 'Count']
sns.barplot(data=language_counts, x='Count', y='Language', hue='Language', legend=False, palette="viridis")
plt.title("Top 15 Languages with Most Netflix Content")
plt.xlabel("Number of Titles")
plt.ylabel("Language")
plt.show()

# OBJECTIVE 2: Content Release Trend Over Time
release_trend = df['Release Year'].value_counts().sort_index()
plt.figure(figsize=(12, 6))
sns.lineplot(x=release_trend.index, y=release_trend.values, marker='o', linewidth=2.5, color='teal')
plt.title("Netflix Content Release Trend Over Years")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles Released")
plt.grid(True)
plt.show()

# OBJECTIVE 3: Optimal Content Duration Analysis
plt.figure(figsize=(10, 6))
sns.boxplot(x='Content Type', y='Hours Viewed', data=df, hue='Content Type', palette='pastel', legend=False)
plt.title("Content Duration Distribution by Type")
plt.xlabel("Content Type")
plt.ylabel("Hours Viewed")
plt.show()

# Histogram of Hours Viewed
plt.figure(figsize=(10, 6))
sns.histplot(df['Hours Viewed'].dropna(), bins=50, kde=True, color='salmon')
plt.title("Distribution of Hours Viewed")
plt.xlabel("Hours Viewed")
plt.ylabel("Frequency")
plt.show()

# OBJECTIVE 4: Multi-language Availability
multi_lang = df['Language Indicator'].value_counts().head(10).reset_index()
multi_lang.columns = ['Language', 'Count']
plt.figure(figsize=(10, 6))
sns.barplot(data=multi_lang, x='Count', y='Language', hue='Language', legend=False, palette="coolwarm")
plt.title("Top Languages with Most Content")
plt.xlabel("Number of Titles")
plt.ylabel("Language")
plt.show()

# OBJECTIVE 5: Top Content by Hours Viewed
top_content = df[['Title', 'Hours Viewed']].dropna().sort_values(by='Hours Viewed', ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x='Hours Viewed', y='Title', data=top_content, hue='Title', palette='mako', legend=False)
plt.title("Top 10 Netflix Titles by Hours Viewed")
plt.xlabel("Hours Viewed")
plt.ylabel("Title")
plt.show()
