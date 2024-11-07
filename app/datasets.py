import pandas as pd
import re

hiragana = re.compile('[\u3040-\u309F]')
katakana = re.compile('[\u30A0-\u30FF]')
CJK = re.compile('[\u4E00-\u9FFF]')

def remove_japanese_characters(text):
    text = hiragana.sub('', text)
    text = katakana.sub('', text)
    text = CJK.sub('', text)
    return text

def truncate_to_300_words(text):
    words = text.split()
    return ' '.join(words[:300])

def count_words(text):
    return len(text.split())

def get_dataset(path, sample_size = 1000):
    df = pd.read_json(path).T.sample(sample_size)

    pattern = re.compile(r"[^\w\s]")
    df['description_cleaned'] = df['description'].apply(lambda desc: remove_japanese_characters(pattern.sub('', desc).lower()))
    df['description_cleaned'] = df['description_cleaned'].apply(truncate_to_300_words)
    df = df[df['description_cleaned'].str.strip().apply(lambda x: count_words(x) >= 50)]
    df = df[df['description_cleaned'].str.strip() != ""]

    return df