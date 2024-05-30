from flask import Flask, render_template, request, flash, redirect, url_for
import secrets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import pdfplumber
import docx

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Generate a random secret key

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Summarization pipeline using BERT
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Function to extract text from a .pdf file
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
        text = ' '.join(pages)
    return text

# Function to extract text from a .docx file
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to preprocess the text
def preprocess_text(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    preprocessed_sentences = []
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        words = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
        preprocessed_sentences.append(' '.join(words))
    return preprocessed_sentences

# Function to generate summary using BERT
def generate_summary(text, num_sentences=3, max_length=1024):
    summaries = []
    for chunk in [text[i:i+max_length] for i in range(0, len(text), max_length)]:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return ' '.join(summaries)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash("No file uploaded", "error")
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('index'))

    flash(f"Uploading document: {file.filename}", "info")

    if file.filename.endswith('.txt'):
        text = file.read().decode('utf-8')
    elif file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(file)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(file)
    else:
        flash("Unsupported file type", "error")
        return redirect(url_for('index'))

    flash(f"Uploaded document: {file.filename}", "success")

    return render_template('index.html', text=text)

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form.get('text')
    if not text:
        flash("No text provided", "error")
        return redirect(url_for('index'))

    flash("Summarizing document...", "info")

    # Redirect to a route that performs the summarization
    return redirect(url_for('generate_summary_route', text=text))

@app.route('/generate_summary')
def generate_summary_route():
    text = request.args.get('text')
    if not text:
        flash("No text provided for summarization", "error")
        return redirect(url_for('index'))

    summary = generate_summary(text)
    flash("Summarization completed", "success")

    return render_template('summary.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
