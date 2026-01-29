# 🧠 LinguaLab – Interactive NLP Explorer

LinguaLab is an interactive Streamlit-based dashboard designed to demonstrate and visualize core **Natural Language Processing (NLP)** concepts in a simple and practical way.

The application allows users to experiment with text data and understand how different NLP techniques work through interactive controls and visual outputs.

---

## 🚀 Features
- Sentence and word tokenization
- Advanced tokenization (blank line, whitespace, wordpunct)
- Stemming using Porter Stemmer
- Lemmatization using WordNet
- Stopwords removal
- N-gram analysis (bigrams, trigrams, n-grams)
- WordCloud visualization
- Part-of-Speech (POS) tagging
- Named Entity Recognition (NER)
- Text vectorization using:
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec
- Interactive sliders and tabs for real-time experimentation

---

## 🛠 Tech Stack
- Python
- Streamlit
- NLTK
- Scikit-learn
- Gensim
- Pandas
- Matplotlib
- WordCloud

---

## 🌐 Live Demo
https://lingualab-nlp-explorer.streamlit.app/


---

## ▶️ How to Run the Project Locally
```bash
pip install -r requirements.txt
streamlit run app.py


📁 Project Structure

LinguaLab-NLP-Explorer/
├── app.py
├── requirements.txt
└── README.md


