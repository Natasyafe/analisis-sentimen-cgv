import pandas as pd
import joblib
import numpy as np
from flask import Flask, render_template, request
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from io import BytesIO
import base64
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools

app = Flask(__name__)

# baca dataset
file_path = 'sentimen_absa.csv'  
data = pd.read_csv(file_path)

#baca model
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
nb_model = joblib.load('naive_bayes_model.pkl')
svm_model = joblib.load('svm_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')

lexicon_positive_df = pd.read_excel('kamus_positive.xlsx')
lexicon_negative_df = pd.read_excel('kamus_negative.xlsx')

lexicon_positive_df.columns = lexicon_positive_df.columns.str.lower().str.strip()
lexicon_negative_df.columns = lexicon_negative_df.columns.str.lower().str.strip()

lexicon_positive_df = lexicon_positive_df.drop_duplicates(subset='word')
lexicon_negative_df = lexicon_negative_df.drop_duplicates(subset='word')

lexicon_positive_dict = dict(zip(lexicon_positive_df['word'], lexicon_positive_df['weight']))
lexicon_negative_dict = dict(zip(lexicon_negative_df['word'], lexicon_negative_df['weight']))

aspect_keywords = {
    'UI/UX': ['tampilan', 'interface', 'desain', 'navigasi', 'user', 'layout'],
    'Pembayaran': ['bayar', 'pembayaran', 'e-wallet', 'dana', 'ovo', 'gopay', 'transaksi'],
    'Promo': ['promo', 'diskon', 'potongan', 'voucher', 'kupon'],
    'Pemesanan': ['pesan', 'booking', 'kursi', 'tiket'],
    'Layanan': ['cs', 'layanan', 'customer', 'respon', 'bantuan', 'komplain']
}

#ABSA
def aspect_sentiment_analysis(text, lexicon_pos, lexicon_neg, aspects_dict):
    aspect_sentiments = {}
    tokens = text.split()

    for aspect, keywords in aspects_dict.items():
        if any(keyword in tokens for keyword in keywords):
            score = 0
            for word in tokens:
                if word in lexicon_pos:
                    score += lexicon_pos[word]
                elif word in lexicon_neg:
                    score += lexicon_neg[word]

            if score > 0:
                label = 'Positif'
            elif score < 0:
                label = 'Negatif'
            else:
                label = 'Netral'

            aspect_sentiments[aspect] = {'score': score, 'sentiment': label}

    return aspect_sentiments

#sentimen lexicon
def sentiment_analysis_lexicon_indonesia(text):
    if not isinstance(text, str):  
        return 0, 'Netral'  

    score = 0
    for word in text.split():
        if word in lexicon_positive_dict:
            score += lexicon_positive_dict[word]
    for word in text.split():
        if word in lexicon_negative_dict:
            score += lexicon_negative_dict[word]

    sentimen = ''
    if score > 0:
        sentimen = 'Positif'
    elif score < 0:
        sentimen = 'Negatif'
    else:
        sentimen = 'Netral'
    return score, sentimen

# Fungsi untuk membandingkan sentimen antar model (NB, SVM, RF)
def compare_model_sentiments():
    df = pd.read_csv('sentimen_absa.csv')

    # Jika kolom prediksi model belum ada, buat prediksi dulu
    if 'Sentimen_NB' not in df.columns:
        df['Sentimen_NB'] = nb_model.predict(tfidf_vectorizer.transform(df['processed_komentar']))
    if 'Sentimen_SVM' not in df.columns:
        df['Sentimen_SVM'] = svm_model.predict(tfidf_vectorizer.transform(df['processed_komentar']))
    if 'Sentimen_RF' not in df.columns:
        df['Sentimen_RF'] = rf_model.predict(tfidf_vectorizer.transform(df['processed_komentar']))

    models = ['NB', 'SVM', 'RF']
    sentiments = ['Positif', 'Negatif', 'Netral']
    color_map = {'Positif': 'green', 'Negatif': 'red', 'Netral': 'blue'}
    
    result_data = []
    for model in models:
        for sentiment in sentiments:
            count = (df[f'Sentimen_{model}'] == sentiment).sum()
            result_data.append({'Model': model, 'Sentimen': sentiment, 'Jumlah': count})

    result_df = pd.DataFrame(result_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=result_df, x='Model', y='Jumlah', hue='Sentimen', palette=color_map)
    plt.title('Perbandingan Sentimen antar Model')
    plt.xlabel('Model')
    plt.ylabel('Jumlah Sentimen')
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Fungsi untuk membuat confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    # Menghitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Menentukan label dan colormap untuk setiap model
    labels = ['Positif', 'Negatif', 'Netral']
    cmap_dict = {
        'Naive Bayes': 'Blues',
        'SVM': 'Reds',
        'Random Forest': 'Greens'
    }
    cmap = cmap_dict.get(model_name, 'Purples')  # Default colormap jika model tidak dikenali

    # Membuat plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=labels, yticklabels=labels, cbar=False)
    
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()

    # Menyimpan gambar ke dalam format PNG
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Fungsi untuk menghitung akurasi
def calculate_accuracy(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred) * 100, 2)

# Fungsi untuk menghasilkan classification report
def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Positif', 'Negatif', 'Netral'])
    return report

# Fungsi untuk menghitung akurasi
def calculate_accuracy(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred) * 100, 2)

# Fungsi untuk membuat grafik distribusi sentimen
def plot_sentiment_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Sentimen', palette='Set2')
    plt.title('Distribusi Sentimen')
    plt.xlabel('Sentimen')
    plt.ylabel('Jumlah')
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

# Fungsi untuk membuat grafik tren sentimen
def plot_sentiment_trend(df):
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        sentiment_trend = df.groupby(df['timestamp'].dt.to_period('M'))['Sentimen'].value_counts().unstack().fillna(0)
        sentiment_trend.plot(kind='line', figsize=(10, 6), marker='o')
        plt.title('Tren Sentimen dari Waktu ke Waktu')
        plt.xlabel('Bulan')
        plt.ylabel('Jumlah Sentimen')
        plt.tight_layout()
        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        return base64.b64encode(img.getvalue()).decode('utf8')
    else:
        return None

def create_wordcloud(df, sentiment):
    text = ' '.join(df[df['Sentimen'] == sentiment]['processed_komentar'])

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if sentiment == 'Positif':
            return 'green'
        elif sentiment == 'Negatif':
            return 'red'
        elif sentiment == 'Netral':
            return 'blue'
        else:
            return 'gray'

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        color_func=color_func,
        collocations=False
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'WordCloud - Sentimen {sentiment}', fontsize=16)
    plt.tight_layout()
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/')
def index():
    page_num = request.args.get('page', default=1, type=int)

    sentimen_filter = request.args.get('sentimen_filter', 'all')

    items_per_page = 10
    start_idx = (page_num - 1) * items_per_page
    end_idx = start_idx + items_per_page

    if sentimen_filter != 'all':
        filtered_data = data[data['Sentimen'].str.lower() == sentimen_filter.lower()]
    else:
        filtered_data = data

    paginated_data = filtered_data.iloc[start_idx:end_idx]

    total_pages = len(filtered_data) // items_per_page + (1 if len(filtered_data) % items_per_page > 0 else 0)

    return render_template('index.html',
                           table_data=paginated_data.to_dict(orient='records'),
                           page=page_num,
                           total_pages=total_pages,
                           selected_sentimen=sentimen_filter)

@app.route('/visualize', methods=['GET'])
def visualize():
    df = pd.read_csv('sentimen_absa.csv')

    # Bagi data ke train/test (tanpa melatih ulang)
    from sklearn.model_selection import train_test_split

    X = df['processed_komentar']
    y = df['Sentimen']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Transformasi test data
    X_test_vec = tfidf_vectorizer.transform(X_test)

    # Prediksi dengan model yang sudah diload
    y_pred_nb = nb_model.predict(X_test_vec)
    y_pred_svm = svm_model.predict(X_test_vec)
    y_pred_rf = rf_model.predict(X_test_vec)

    # Plot confusion matrix dan akurasi untuk test set
    cm_nb = plot_confusion_matrix(y_test, y_pred_nb, 'Naive Bayes')
    cm_svm = plot_confusion_matrix(y_test, y_pred_svm, 'SVM')
    cm_rf = plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest')

    acc_nb = calculate_accuracy(y_test, y_pred_nb)
    acc_svm = calculate_accuracy(y_test, y_pred_svm)
    acc_rf = calculate_accuracy(y_test, y_pred_rf)

    # Generate classification report untuk setiap model
    report_nb = generate_classification_report(y_test, y_pred_nb)
    report_svm = generate_classification_report(y_test, y_pred_svm)
    report_rf = generate_classification_report(y_test, y_pred_rf)

    # Visualisasi lainnya tetap bisa pakai seluruh data
    sentiment_distribution_img = plot_sentiment_distribution(df)
    sentiment_trend_img = plot_sentiment_trend(df)
    wordcloud_pos = create_wordcloud(df, 'Positif')
    wordcloud_neg = create_wordcloud(df, 'Negatif')
    wordcloud_neutral = create_wordcloud(df, 'Netral')
    model_comparison_img = compare_model_sentiments()

    return render_template('visualize.html', 
                           sentiment_distribution_img=sentiment_distribution_img,
                           sentiment_trend_img=sentiment_trend_img,
                           wordcloud_pos=wordcloud_pos,
                           wordcloud_neg=wordcloud_neg,
                           wordcloud_neutral=wordcloud_neutral,
                           model_comparison_img=model_comparison_img,
                           cm_nb=cm_nb, cm_svm=cm_svm, cm_rf=cm_rf,
                           acc_nb=acc_nb, acc_svm=acc_svm, acc_rf=acc_rf,
                           report_nb=report_nb, report_svm=report_svm, report_rf=report_rf)

if __name__ == '__main__':
    app.run(debug=True)
