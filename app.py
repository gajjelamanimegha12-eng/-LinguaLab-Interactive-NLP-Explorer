import streamlit as st
import nltk
import re
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    WhitespaceTokenizer,
    WordPunctTokenizer,
    blankline_tokenize
)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
from wordcloud import WordCloud

# ---------------- NLTK DOWNLOADS ----------------
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")   # 🔥 THIS LINE FIXES YOUR ERROR
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("maxent_ne_chunker")
nltk.download("words")


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="LinguaLab – Interactive NLP Explorer",
    page_icon="🧠",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧠 NLP Configuration Panel")

st.sidebar.markdown("### 🔹 Tokenization")
min_word_len = st.sidebar.slider(
    "Minimum Word Length Filter",
    2, 8, 3
)

st.sidebar.markdown("### 🔹 N-gram Analysis")
ngram_n = st.sidebar.slider(
    "N-gram Size (Bi / Tri / N-gram)",
    2, 5, 2
)

st.sidebar.markdown("### 🔹 WordCloud")
max_wc_words = st.sidebar.slider(
    "Maximum Words in WordCloud",
    20, 200, 100
)

st.sidebar.markdown("### 🔹 Bag of Words (BoW)")
bow_features = st.sidebar.slider(
    "BoW Feature Limit",
    50, 1000, 300, step=50
)

st.sidebar.markdown("### 🔹 TF-IDF")
tfidf_features = st.sidebar.slider(
    "TF-IDF Feature Limit",
    50, 1000, 300, step=50
)

st.sidebar.markdown("### 🔹 Word2Vec")
w2v_size = st.sidebar.slider(
    "Vector Dimension Size",
    20, 100, 50, step=10
)
w2v_window = st.sidebar.slider(
    "Context Window Size",
    2, 10, 5
)

st.sidebar.markdown("---")
st.sidebar.caption("✨ Interactive NLP Learning Dashboard")

# ---------------- MAIN HEADER ----------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 LinguaLab – Interactive NLP Explorer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Understanding Text from Tokens to Meaning</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- INPUT ----------------
text = st.text_area(
    "✍️ Enter Text for Analysis",
    height=200,
    placeholder="Paste a paragraph, article, or any text here..."
)
st.caption("💡 Tip: Longer text gives richer NLP insights")

# ---------------- RUN NLP ----------------
if st.button("🚀 Run NLP Pipeline"):

    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # ---------------- CLEANING ----------------
        clean_text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower()
        clean_text = re.sub(r'\s+', ' ', clean_text)

        stop_words = set(stopwords.words("english"))

        sentences = sent_tokenize(clean_text)
        words = word_tokenize(clean_text)

        words_no_stop = [
            w for w in words
            if w not in stop_words and len(w) >= min_word_len
        ]

        # ---------------- STEMMING ----------------
        stemmer = PorterStemmer()
        stemmed_words = [stemmer.stem(w) for w in words_no_stop]

        # ---------------- LEMMATIZATION ----------------
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(w) for w in words_no_stop]

        # ---------------- METRICS ----------------
        c1, c2, c3 = st.columns(3)
        c1.metric("📄 Sentences", len(sentences))
        c2.metric("🔤 Total Words", len(words))
        c3.metric("📚 Unique Vocabulary", len(set(words_no_stop)))

        st.markdown("---")

        # ---------------- TABS ----------------
        tabs = st.tabs([
            "📘 Text Tokenization",
            "✂️ Stemming Analysis",
            "📖 Lemmatization Analysis",
            "🧩 Advanced Tokenizers",
            "🔗 N-gram Analysis",
            "☁️ WordCloud Visualization",
            "🏷️ POS Tagging & NER",
            "📊 Vector Representations"
        ])

        # -------- TOKENIZATION --------
        with tabs[0]:
            st.subheader("Sentence Tokenization")
            st.write(sentences)

            st.subheader("Word Tokenization (Filtered)")
            st.write(words_no_stop)

        # -------- STEMMING --------
        with tabs[1]:
            st.subheader("Porter Stemmer Output")
            st.write(stemmed_words)

        # -------- LEMMATIZATION --------
        with tabs[2]:
            st.subheader("WordNet Lemmatizer Output")
            st.write(lemmatized_words)

        # -------- ADVANCED TOKENIZATION --------
        with tabs[3]:
            st.subheader("Blank Line Tokenization")
            st.write(blankline_tokenize(text))

            st.subheader("Whitespace Tokenization")
            st.write(WhitespaceTokenizer().tokenize(text))

            st.subheader("WordPunct Tokenization")
            st.write(WordPunctTokenizer().tokenize(text))

        # -------- N-GRAMS --------
        with tabs[4]:
            st.subheader(f"{ngram_n}-gram Analysis")
            cv_ngram = CountVectorizer(
                ngram_range=(ngram_n, ngram_n),
                max_features=bow_features
            )
            ngram_df = pd.DataFrame(
                cv_ngram.fit_transform([clean_text]).toarray(),
                columns=cv_ngram.get_feature_names_out()
            )
            st.dataframe(ngram_df, use_container_width=True)

        # -------- WORDCLOUD --------
        with tabs[5]:
            wc_text = " ".join(lemmatized_words)
            wc = WordCloud(
                width=1000,
                height=420,
                background_color="white",
                max_words=max_wc_words,
                colormap="inferno"
            ).generate(wc_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

        # -------- POS & NER --------
        with tabs[6]:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Part-of-Speech Tagging")
                pos_df = pd.DataFrame(
                    pos_tag(words_no_stop),
                    columns=["Word", "POS Tag"]
                )
                st.dataframe(pos_df, use_container_width=True)

            with col2:
                st.subheader("Named Entity Recognition")
                st.write(ne_chunk(pos_tag(words_no_stop)))

        # -------- VECTOR REPRESENTATIONS --------
        with tabs[7]:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Bag of Words (BoW)")
                bow = CountVectorizer(max_features=bow_features)
                bow_df = pd.DataFrame(
                    bow.fit_transform([clean_text]).toarray(),
                    columns=bow.get_feature_names_out()
                )
                st.dataframe(bow_df, use_container_width=True)

            with col2:
                st.subheader("TF-IDF Representation")
                tfidf = TfidfVectorizer(max_features=tfidf_features)
                tfidf_df = pd.DataFrame(
                    tfidf.fit_transform([clean_text]).toarray(),
                    columns=tfidf.get_feature_names_out()
                )
                st.dataframe(tfidf_df, use_container_width=True)

            st.markdown("---")

            st.subheader("Word2Vec Embeddings")
            tokenized_sentences = [
                [w for w in word_tokenize(s) if w not in stop_words]
                for s in sentences
            ]

            model = Word2Vec(
                tokenized_sentences,
                vector_size=w2v_size,
                window=w2v_window,
                min_count=1
            )

            selected_word = st.selectbox(
                "Select a word to view embedding",
                model.wv.key_to_index.keys()
            )
            st.write(model.wv[selected_word][:10])

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "LinguaLab | Tokenization • Stemming • Lemmatization • NLP Visualization"
)










