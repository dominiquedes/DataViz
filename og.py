from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
import atexit
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set the style for all plots
plt.style.use('seaborn-v0_8')
sns.set_theme(style="whitegrid")
sns.set_palette("husl")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def filter_outliers(df):
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def generate_plot(df, column):
    plt.clf()
    
    # Determine the data type and create appropriate visualization
    if df[column].dtype in ['int64', 'float64']:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram with KDE
        sns.histplot(data=df, x=column, ax=ax1, kde=True, 
                    color='#3498db', edgecolor='black', alpha=0.7)
        ax1.set_title(f'Distribution of "{column}"', fontsize=14, pad=20)
        ax1.set_xlabel(column, fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        
        # Box plot
        sns.boxplot(y=df[column], ax=ax2, color='#2ecc71')
        ax2.set_title(f'Box Plot of "{column}"', fontsize=14, pad=20)
        ax2.set_ylabel(column, fontsize=12)
        
        # Add statistics annotations
        stats_text = f"""
        Mean: {df[column].mean():.2f}
        Median: {df[column].median():.2f}
        Std Dev: {df[column].std():.2f}
        Skewness: {df[column].skew():.2f}
        """
        fig.text(0.99, 0.5, stats_text, fontsize=10, 
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    elif df[column].dtype == 'object':
        value_counts = df[column].value_counts()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar plot for top 10 values
        top_values = value_counts.nlargest(10)
        sns.barplot(x=top_values.values, y=top_values.index, ax=ax1, 
                   palette='viridis', orient='h')
        ax1.set_title(f'Top 10 Most Frequent Values in "{column}"', fontsize=14, pad=20)
        ax1.set_xlabel("Count", fontsize=12)
        ax1.set_ylabel(column, fontsize=12)
        
        # Pie chart for top 5 values
        top_5 = value_counts.nlargest(5)
        ax2.pie(top_5.values, labels=top_5.index, autopct='%1.1f%%',
                colors=sns.color_palette('pastel'), startangle=90)
        ax2.set_title(f'Top 5 Values Distribution', fontsize=14, pad=20)
        
        # Add statistics annotations
        stats_text = f"""
        Total Unique Values: {len(value_counts)}
        Most Common: {value_counts.index[0]}
        Least Common: {value_counts.index[-1]}
        """
        fig.text(0.99, 0.5, stats_text, fontsize=10, 
                verticalalignment='center', bbox=dict(facecolor='white', alpha=0.8))

    else:
        return None

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=300)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def perform_pca(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty:
        return None

    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(numeric_df)

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_df)

    plt.clf()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot with density
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                        alpha=0.6, c='#3498db', s=50)
    
    # Add explained variance ratio to the title
    explained_var = pca.explained_variance_ratio_
    ax.set_title(f"PCA Projection\nExplained Variance: {explained_var[0]:.1%} + {explained_var[1]:.1%}", 
                fontsize=14, pad=20)
    
    ax.set_xlabel(f"Principal Component 1 ({explained_var[0]:.1%})", fontsize=12)
    ax.set_ylabel(f"Principal Component 2 ({explained_var[1]:.1%})", fontsize=12)
    
    # Add grid and styling
    ax.grid(True, linestyle='--', alpha=0.7)
    sns.despine(ax=ax, offset=10)
    
    # Add colorbar for density
    plt.colorbar(scatter, label='Density')

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def perform_clustering(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None

    # Determine optimal number of clusters using elbow method
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(numeric_df)
        distortions.append(kmeans.inertia_)

    # Find the elbow point
    elbow_point = np.argmin(np.diff(distortions)) + 1
    optimal_k = min(max(2, elbow_point), 5)  # Limit between 2 and 5 clusters

    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot with clusters
    scatter = ax1.scatter(
        numeric_df.iloc[:, 0],
        numeric_df.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.7,
        s=50
    )
    ax1.set_title(f"KMeans Clustering ({optimal_k} Groups)", fontsize=14, pad=20)
    ax1.set_xlabel(numeric_df.columns[0], fontsize=12)
    ax1.set_ylabel(numeric_df.columns[1], fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Elbow plot
    ax2.plot(K, distortions, 'bx-')
    ax2.plot(optimal_k, distortions[optimal_k-1], 'ro', markersize=10)
    ax2.set_title('Elbow Method for Optimal K', fontsize=14, pad=20)
    ax2.set_xlabel('Number of clusters (K)', fontsize=12)
    ax2.set_ylabel('Distortion', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def generate_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None

    plt.clf()
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr = numeric_df.corr()
    
    # Create heatmap
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, 
                fmt='.2f', ax=ax)
    
    ax.set_title('Correlation Heatmap', fontsize=14, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png', dpi=300)
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')

    try:
        df = pd.read_csv(file)
        df = filter_outliers(df)
        
        stats = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict()
        }

        plots = {}
        for column in df.columns:
            plot = generate_plot(df, column)
            if plot:
                plots[column] = plot

        pca_plot = perform_pca(df)
        clustering_plot = perform_clustering(df)
        correlation_plot = generate_correlation_heatmap(df)

        return render_template(
            'results.html',
            stats=stats,
            plots=plots,
            pca_plot=pca_plot,
            clustering_plot=clustering_plot,
            correlation_plot=correlation_plot
        )

    except Exception as e:
        print("Error processing file:", str(e))
        return render_template('index.html', error=str(e))


@atexit.register
def cleanup_on_shutdown():
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        for file in os.listdir(app.config['UPLOAD_FOLDER']):
            try:
                os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
            except OSError:
                pass

if __name__ == '__main__':
    app.run(debug=True)