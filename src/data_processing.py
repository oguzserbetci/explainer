import re

def tokenize(sample, tokenizer=None):
    sample['labels'] = sample['sentiment']
    return tokenizer(sample['text'], truncation=True, max_length=512)


def tokenize_with_overflow(sample, tokenizer=None):
    return tokenizer(sample['text'], truncation=True, max_length=512, return_overflowing_tokens=True, padding=True)


def overlapping_spans(text, tokenizer, max_length, stride=20):
    input_ids = tokenizer(
        text,
        return_overflowing_tokens=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
    )["input_ids"]
    decoded = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    return decoded


def get_sentences(text, sections, sentecizer):
    sentecizer = SoMaJo("de_CMC")
    
    sents = []
    for section_text, section in zip(text.split("\n"), sections):
        for sent in sentecizer.tokenize_text([section_text.strip()]):
            sent = " ".join([token.text for token in sent])
            section = re.split("(-|_|text)", section, flags=re.IGNORECASE)[0].strip()
            if len(sent) > 400:
                sents.extend(
                    [
                        f"{section}: {span}"
                        for span in overlapping_spans(section_text.strip(), 128, 20)
                    ]
                )
            else:
                sents.append(f"{section}: {sent}")
    return sents

def truncate(sample, tokenizer):
    return tokenizer(
        sample["text"],
        truncation=True,
        return_tensors="pt",
        padding=True,
    )

def sent(sample, tokenizer):
    return tokenizer(
        get_sentences(sample["text"], sample["text_fields"]),
        truncation=True,
        return_tensors="pt",
        padding=True,
    )

def max_len_200(sample, tokenizer):
    return tokenizer(
        sample["text"],
        truncation=True,
        max_length=200,
        return_tensors="pt",
        padding=True,
    )