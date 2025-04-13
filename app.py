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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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
    fig, ax = plt.subplots(figsize=(8, 5))

    if df[column].dtype in ['int64', 'float64']:
        sns.histplot(data=df, x=column, ax=ax, kde=True, color='skyblue', edgecolor='black')
        ax.set_title(f'Distribution of "{column}"', fontsize=14)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)

    elif df[column].dtype == 'object':
        value_counts = df[column].value_counts().nlargest(10)
        if len(value_counts) > 1:
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax, palette='muted')
            ax.set_title(f'Top 10 Most Frequent Values in "{column}"', fontsize=14)
            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
        else:
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.set_title(f'Pie Chart of "{column}"', fontsize=14)

    else:
        return None

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
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
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7, c='royalblue')
    ax.set_title("PCA Projection (2 Components)", fontsize=14)
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.grid(True)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode()

def perform_clustering(df):
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(numeric_df)

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        numeric_df.iloc[:, 0],
        numeric_df.iloc[:, 1],
        c=clusters,
        cmap='viridis',
        alpha=0.7
    )
    ax.set_title("KMeans Clustering (3 Groups)", fontsize=14)
    ax.set_xlabel(numeric_df.columns[0], fontsize=12)
    ax.set_ylabel(numeric_df.columns[1], fontsize=12)
    ax.grid(True)

    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
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
            'missing_values': df.isnull().sum().to_dict()
        }

        plots = {}
        for column in df.columns:
            plot = generate_plot(df, column)
            if plot:
                plots[column] = plot

        pca_plot = perform_pca(df)
        clustering_plot = perform_clustering(df)

        return render_template(
            'results.html',
            stats=stats,
            plots=plots,
            pca_plot=pca_plot,
            clustering_plot=clustering_plot
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
