import logging
import re

from nltk.corpus import stopwords
from somajo import SoMaJo

ENGLISH_STOPWORDS = stopwords.words("english")
GERMAN_STOPWORDS = stopwords.words("german")

SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]"}
PUNCTUATION = set(".;:'\".,!?-_=+§()[]/\\%#&")

GERMAN_STOPWORD_MOD = {
    "add": ["hätte", "wäre"],
    "remove": [
        "kann",
        "kein",
        "keine",
        "keinem",
        "keinen",
        "keiner",
        "keines",
        "können",
        "könnte",
        "nicht",
        "nur",
        "noch",
        "sehr",
        "selbst",
        "sich",
        "soll",
        "sollte",
        "sonst",
        "sondern",
        "viel",
        "will",
        "wollen",
        "wollte",
        "doch",
        "durch",
    ],
}

for word in GERMAN_STOPWORD_MOD["add"]:
    GERMAN_STOPWORDS.append(word)
for word in GERMAN_STOPWORD_MOD["remove"]:
    GERMAN_STOPWORDS.remove(word)


def filter_words(word, stopwords=None):
    _special_tokens = {"[SEP]", "[CLS]", "[PAD]"}
    _punctuation = set(".;:'\".,!?-_=+§()[]/\\%#&")
    return (
        (word not in _special_tokens)
        and (word not in _punctuation)
        and ((stopwords is None) or (word.lower() not in stopwords))
    )


def get_attrs(word_attributions, stopwords=None, eliminate=False):
    simpler_list = []
    last_word = None
    attrs = []
    SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]"}

    def should_include_word(word):
        if not word or word is None or word in SPECIAL_TOKENS:
            return False
        if not eliminate:
            return True
        return filter_words(word, stopwords)

    for word, attr in word_attributions:
        if word.startswith("##"):
            last_word += word.removeprefix("##")
            attrs.append(attr)
        else:
            if should_include_word(last_word):
                simpler_list.append((last_word, sum(attrs)))
                # simpler_list.append((last_word, max(attrs, key=abs)))
            last_word = word
            attrs = [attr]

    # Handle the final word
    if should_include_word(last_word):
        simpler_list.append((last_word, sum(attrs)))
        # simpler_list.append((last_word, max(attrs, key=abs)))

    return simpler_list


def get_ngram_attrs(data, min_n, max_n):
    ngrams_list = []
    for n in range(min_n, max_n + 1):
        for i in range(len(data) - n + 1):
            ngram = data[i : i + n]
            ngram_words = " ".join([word[0] for word in ngram]).lower()
            attribution = sum([word[1] for word in ngram]) / n
            ngrams_list.append((ngram_words, attribution, f"{n}gram"))
    return ngrams_list


def get_sentences(text, tokenizer=None, sentence_model: SoMaJo = None):
    if tokenizer is not None:
        text = tokenizer.convert_tokens_to_string(
            tokenizer.batch_decode(
                tokenizer(text)["input_ids"], clean_up_tokenization_spaces=True, skip_special_tokens=True
            )
        )
    if sentence_model is not None:
        return [
            "".join([token.text.replace(" ", "") for token in sent])
            for sent in sentence_model.tokenize_text(re.split(r"\n|<\s*?br\s*?/\s*?>", text))
        ]
    else:
        return re.split(r".|\n|<\s*?br\s*?/\s*?>", text)


def get_sentence_ngram_attrs(sentences, all_attrs, stopwords, eliminate=False):
    sentence_attrs = []
    ngram_attrs = []

    if not sentences:
        return sentence_attrs, ngram_attrs

    current_sentence = sentences.pop(0).lower()
    uptillnow = []
    uptillnowscores = []
    for word, attr in all_attrs:
        word = word.replace(" ", "").strip().lower()
        current_match = "".join(uptillnow + [word])

        if not current_sentence.startswith(current_match):
            # Reset and try this word as the start of a new match
            uptillnow = []
            uptillnowscores = []
            continue

        uptillnow.append(word)
        uptillnowscores.append(attr)

        if current_match == current_sentence:
            sentence_attrs.append(
                (" ".join(uptillnow).lower(), sum(uptillnowscores) / len(uptillnowscores), "sentence")
            )
            filtered_ = [
                (w, a) for w, a in zip(uptillnow, uptillnowscores) if not eliminate or filter_words(w, stopwords)
            ]
            ngram_attrs.extend(get_ngram_attrs(filtered_, 1, 3))
            uptillnow = []
            uptillnowscores = []
            if sentences:
                current_sentence = sentences.pop(0).lower()
            else:
                break

    if not sentence_attrs or sentences:
        filtered_ = [(w, a) for w, a in all_attrs if not eliminate or filter_words(w, stopwords)]
        ngram_attrs = get_ngram_attrs(filtered_, 1, 3)
    elif uptillnow:
        sentence_attrs.append((" ".join(uptillnow).lower(), sum(uptillnowscores) / len(uptillnowscores), "rest"))
        filtered_ = [(w, a) for w, a in zip(uptillnow, uptillnowscores) if not eliminate or filter_words(w, stopwords)]
        ngram_attrs.extend(get_ngram_attrs(filtered_, 1, 3))

    return sentence_attrs, ngram_attrs


def get_attributions(word_attributions, sentences=None, language=None):
    if language == "en":
        stopwords = ENGLISH_STOPWORDS
    elif language == "de":
        stopwords = GERMAN_STOPWORDS
    else:
        stopwords = None

    all_attrs = get_attrs(word_attributions, stopwords, eliminate=False)
    if not all_attrs:
        raise ValueError("no attributions")

    if sentences is not None:
        sent_attrs, ngram_attrs = get_sentence_ngram_attrs(list(sentences), all_attrs, stopwords, eliminate=True)
    else:
        filtered_ = [(w, a) for w, a in all_attrs if filter_words(w, stopwords)]
        ngram_attrs = get_ngram_attrs(filtered_, 1, 3)
        sent_attrs = []

    if (sentences is not None) and (len(sent_attrs) != len(sentences)):
        logging.warning(f"{len(sent_attrs)}, {len(sentences)}")

    if ngram_attrs:
        terms, attrs, types = zip(*ngram_attrs)
    else:
        raise ValueError("no attributions")

    if sent_attrs:
        sent_terms, sent_attrs, sent_types = zip(*sent_attrs)
        terms += sent_terms
        attrs += sent_attrs
        types += sent_types

    return terms, attrs, types
