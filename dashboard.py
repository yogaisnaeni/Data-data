# Masukkan semua kode Anda di sini

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import GridSearchCV

# Fungsi untuk visualisasi K-Means Clustering
def kmeans_visualization(df):
    st.subheader("K-Means Clustering")

    # Preprocessing
    numerical_cols = df.select_dtypes(include='number').columns
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    # PCA untuk reduksi dimensi
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(scaled_data)

    # K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_data)
    df['Cluster'] = kmeans.labels_

    # Visualisasi Clustering dengan PCA
    st.write("### Clustering Visualization")
    # Membuat mapping untuk keterangan risiko
    kmeans = KMeans(n_clusters=3, random_state=42)

    # Latih model K-Means
    kmeans.fit(scaled_data)

    risk_mapping = {0: 'Low Risk', 1: 'Medium Risk', 2: 'High Risk'}

    # Akses centroid dari setiap cluster
    centroids = kmeans.cluster_centers_
    custom_palette = {'Low Risk': 'green', 'Medium Risk': 'cyan', 'High Risk': 'red'}
    df['Risk Level'] = df['Cluster'].map(risk_mapping)

    # Visualisasi hasil clustering dengan PCA data
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=df['Risk Level'], palette=custom_palette)

    # Menambahkan centroid ke plot
    centroids_pca = pca.transform(kmeans.cluster_centers_)  # Proyeksikan centroid ke ruang 2D (PCA)
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='o', s=200, c='black', label='Centroids')

    # Menambahkan keterangan legend
    plt.legend(title='Risk Level')

    # Menambahkan judul dan label
    plt.title('K-Means Clustering dengan PCA dan Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    # Menampilkan plot
    st.pyplot(plt)

    # Elbow Method untuk optimal k
    st.write("### Elbow Method")
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(scaled_data)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 10), inertia, marker='o', linestyle='--')
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS")
    st.pyplot(plt)

    # Visualisasi distribusi cluster
    st.write("### Cluster Distribution")
    # Menghitung distribusi risiko
    risk_counts = df['Risk Level'].value_counts()

    # Definisikan warna yang sesuai dengan kategori risiko
    custom_colors = ['cyan', 'red', 'green']

    # Visualisasi distribusi dengan pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=custom_colors)

    # Menambahkan judul
    plt.title('Distribusi Risk Level')
    st.pyplot(plt)

# Fungsi untuk visualisasi Logistic Regression
def logistic_regression_visualization(df):
    st.subheader("Logistic Regression")

    # Preprocessing
    if 'loan_status' not in df.columns:
        st.error("Kolom 'loan_status' tidak ditemukan dalam dataset.")
        return

    # Identifikasi kolom non-numerik
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Lakukan encoding untuk kolom kategorikal
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    # Pilih hanya kolom numerik untuk K-Means
    numerical_cols = df.select_dtypes(include=['number']).drop('loan_status', axis=1).columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Risk_Level'] = kmeans.fit_predict(scaled_data)

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi Logistic Regression
    logreg = LogisticRegression(random_state=42)

    # Melatih model
    logreg.fit(X_train, y_train)

    # Probabilitas Prediksi
    y_probs = logreg.predict_proba(X_test)

    # Probabilitas prediksi saat ini
    y_probs_df = pd.DataFrame(y_probs, columns=logreg.classes_)

    # Hitung rata-rata probabilitas dari semua data
    medium_risk_mean = y_probs_df.mean(axis=1)

    # Tambahkan kolom "Medium Risk" dengan nilai rata-rata
    y_probs_df['Medium Risk'] = medium_risk_mean

    # Urutkan kolom menjadi Low Risk, Medium Risk, High Risk
    # Jika Low Risk dan High Risk sudah ada, tambahkan Medium Risk di antara keduanya
    all_classes = ['Low Risk', 'Medium Risk', 'High Risk']
    for col in all_classes:
        if col not in y_probs_df.columns:
            y_probs_df[col] = 0  # Tambahkan jika kolom tidak ada

    y_probs_df = y_probs_df[all_classes]
    y_probs_df = y_probs_df.div(y_probs_df.sum(axis=1), axis=0)

    # Confusion Matrix
    st.write("### Confusion Matrix")
    y_pred = logreg.predict(X_test)  # Menghasilkan label kelas diskret

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Visualisasi Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=logreg.classes_)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix Logistic Regression')
    st.pyplot(plt)

def roc_auc_visualization(df):
    st.subheader("ROC AUC Analysis")

    if 'loan_status' not in df.columns:
        st.error("Column 'loan_status' not found in dataset.")
        return

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    numerical_cols = df.select_dtypes(include=['number']).drop('loan_status', axis=1).columns

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[numerical_cols])

    X = df.drop('loan_status', axis=1)
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(random_state=42)
    logreg.fit(X_train, y_train)

    y_probs = logreg.predict_proba(X_test)
    y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(y_test_binarized, y_probs, multi_class='ovr', average='macro')

    st.write("### ROC AUC (Before Tuning)")
    st.write(f"ROC AUC Score: {roc_auc:.2f}")

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [100, 200, 300]
    }

    grid_search = GridSearchCV(LogisticRegression(multi_class='ovr'), param_grid, cv=5, scoring='roc_auc_ovr', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_probs_tuned = best_model.predict_proba(X_test)
    roc_auc_tuned = roc_auc_score(y_test_binarized, y_probs_tuned, multi_class='ovr', average='macro')

    st.write("### ROC AUC (After Tuning)")
    st.write(f"ROC AUC Score: {roc_auc_tuned:.2f}")

    st.write("### ROC AUC (Comparison)")
    roc_auc_scores = [roc_auc, roc_auc_tuned]
    labels = ['Before Tuning', 'After Tuning']

    plt.figure(figsize=(8, 6))
    plt.bar(labels, roc_auc_scores, color=['skyblue', 'orange'])
    plt.ylim(0.5, 1.0)
    plt.ylabel('ROC AUC Score')
    plt.title('Comparison of ROC AUC Before and After Tuning')
    st.pyplot(plt)


# Fungsi utama aplikasi
def main():
    st.title("Dashboard Analisis Data")
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Pilih Analisis", ["K-Means Clustering", "Logistic Regression", "ROC AUC Analysis"])

    # Load dataset
    st.sidebar.write("### Upload Dataset")
    uploaded_file = "loan_data.csv"

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Dataset")
        st.dataframe(df.head())
        numerical_cols = df.select_dtypes(include=['number']).drop('loan_status', axis=1).columns
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols])

        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)

        # Jalankan K-Means
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['Cluster'] = kmeans.fit_predict(scaled_data)

        # Tambahkan hasil clustering ke dataframe
        df['Cluster'] = kmeans.labels_

        def remove_outliers_iqr(df, column):
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            initial_count = df.shape[0]
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            final_count = df.shape[0]
            return df, initial_count != final_count

        columns_to_clean = numerical_cols

        outliers_removed = True
        while outliers_removed:
            outliers_removed = False
            for col in columns_to_clean:
                df, removed = remove_outliers_iqr(df, col)
                if removed:
                    outliers_removed = True


        if option == "K-Means Clustering":
            kmeans_visualization(df)
        elif option == "Logistic Regression":
            logistic_regression_visualization(df)
        elif option == "ROC AUC Analysis":
            roc_auc_visualization(df)
    else:
        st.write("Silakan upload dataset untuk memulai analisis.")

if __name__ == '__main__':
    main()
