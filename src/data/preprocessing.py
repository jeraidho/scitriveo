import re
import pandas as pd
import spacy
from langdetect import detect
import nltk
from nltk.corpus import stopwords
from pandarallel import pandarallel

nltk.download('stopwords')


# main class for text preprocessing
class TextPreprocessor:
    additional_stop_words = [
        'study', 'research', 'method', 'result', 'findings', 'paper', 'theory', 'data',
        'approach', 'analysis', 'review', 'article'
    ]

    def __init__(self):
        # get regex filters
        self.html_regex = re.compile(r'<[^>]*>')
        self.url_regex = re.compile(r'https?://\S+|www\.\S+')
        self.spacelike_regex = re.compile(r'\s+')

        # initialise spaCy for English
        self.nlp = spacy.load("en_core_web_sm")

        # stopwords
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update(self.additional_stop_words)

    def _clean_noise(self, text: str) -> str:
        """
        Clean html-tags, url-links, multiple space-like chars
        :param text: text to be cleaned
        :return: preprocessed text
        """
        text = self.html_regex.sub(' ', text)
        text = self.url_regex.sub(' ', text)
        text = self.spacelike_regex.sub(' ', text)
        return text.strip().lower()

    def process_string(self, text: str) -> str:
        if not isinstance(text, str) or not text:
            return ""

        cleaned = self._clean_noise(text)
        if not cleaned:
            return ""

        # spaCy lemmatization
        doc = self.nlp(cleaned)
        lemmatised_text = []

        for token in doc:
            lemma = token.lemma_.lower().strip()

            # skip spaces, punctuation and empty tokens
            if token.is_space or token.is_punct:
                continue

            # skip stopwords
            if lemma in self.stop_words:
                continue

            # skip empty / very short / non-alphabetic tokens
            if not lemma or len(lemma) < 1:
                continue
            if not any(char.isalpha() for char in lemma):
                continue

            lemmatised_text.append(lemma)

        return " ".join(lemmatised_text)

    @staticmethod
    def detect_language(text: str) -> str:
        """ Detect language of the input text """
        try:
            return detect(text)
        except:
            return 'unknown'

    def preprocessing_pipeline(self, input_df: pd.DataFrame) -> pd.DataFrame:
        # init parallel processing
        pandarallel.initialize(progress_bar=False)

        input_df = input_df.copy()

        # Detect language of title and abstract
        input_df['title_lang'] = input_df['title'].apply(self.detect_language)
        input_df['abstract_lang'] = input_df['abstract'].apply(self.detect_language)

        # add column for cleaned and lemmatised title and abstract
        input_df['title_lemmatised'] = input_df['title'].parallel_apply(self.process_string)
        input_df['abstract_lemmatised'] = input_df['abstract'].parallel_apply(self.process_string)

        # create index text based on language
        input_df['index_text_lemmatised'] = input_df.apply(
            lambda row: (
                row['title_lemmatised'] + ' ' + row['abstract_lemmatised']
                if row['title_lang'] == 'en' else row['abstract_lemmatised']
            ),
            axis=1
        )

        # clean empty title and abstract
        input_df = input_df[
            ~(
                    (input_df['title_lemmatised'] == '') &
                    (input_df['abstract_lemmatised'] == '')
            )
        ]

        input_df = input_df.drop(columns=['title_lang', 'abstract_lang'])

        return input_df.reset_index(drop=True)


if __name__ == '__main__':
    file_path = 'abstracts.csv'
    data = pd.read_csv(file_path)

    text_preprocessor = TextPreprocessor()
    processed_data = text_preprocessor.preprocessing_pipeline(data)

    processed_data.to_csv('abstracts_lemmatized.csv', index=False)
