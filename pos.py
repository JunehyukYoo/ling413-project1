# NOTE This file is not being run as part of main.py right now.
from collections import defaultdict, Counter
import pandas as pd
import pyconll
from sklearn.metrics import classification_report, confusion_matrix
from spacy.tokens import Doc

def train_majority_baseline_pos(conll_file):
    '''
    Load UD corpus in CoNLL-U format.
    Returns trained model (feature vectors).
    '''
    counts = defaultdict(Counter)
    for sentence in conll_file:
        for word in sentence:
            counts[word.form.lower()][word.upos] += 1

    # word -> most frequent POS tag
    baseline_model = {word: pos_counts.most_common(1)[0][0]
                      for word, pos_counts in counts.items()}
    
    return baseline_model

def predict_majority_baseline_pos(model, sentence):
    '''
    Predict POS tags for one sentence
    '''
    predictions = []
    for word in sentence:
        predictions.append(model.get(word.form.lower(), "NOUN"))  # default fallback
    return predictions

def evaluate_baseline_pos(model, conll_file):
    '''
    Evaluate the baseline model on a test set.
    '''
    actual, prediction = [], []
    for sentence in conll_file:
        actual.extend([word.upos for word in sentence])
        prediction.extend(predict_majority_baseline_pos(model, sentence))
        
    report = classification_report(actual, prediction)
    return report

# def evaluate_spacy(model, conll_file):
#     '''
#     Evaluate the baseline model on a test set.
#     '''
#     actual, prediction = [], []
#     for sentence in conll_file:
#         words = [word.form for word in sentence]
#         actual.extend([word.upos for word in sentence])
#         text = " ".join(word.form for word in sentence)
#         doc = Doc(model.vocab, words=words)
#         doc = model(doc)
#         prediction.extend([token.pos_ for token in doc])
        
#     report = classification_report(actual, prediction)
#     return report

def normalize_ud_tag(tag):
    '''
    Normalize Spacy and UD POS tags to a common set of tags. 
    Specifically, map UD's AUX, INTJ, SYM tags to X as spaCy doesnt include them.
    '''
    if tag in {"AUX", "INTJ", "SYM"}:
        return "X"
    return tag

def evaluate_spacy_pos(model, conll_file):
    actual, prediction = [], []
    for sentence in conll_file:
        words = [w.form for w in sentence]
        actual_tags = [normalize_ud_tag(w.upos) for w in sentence]

        # Create spaCy doc with UD tokenization
        doc = Doc(model.vocab, words=words)
        doc = model(doc)

        predicted_tags = [normalize_ud_tag(tok.pos_) for tok in doc]

        assert len(actual_tags) == len(predicted_tags)
        actual.extend(actual_tags)
        prediction.extend(predicted_tags)
    report = classification_report(actual, prediction)

    # Confusion matrix as DataFrame
    labels = sorted(set(actual) | set(prediction))
    cm = confusion_matrix(actual, prediction, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    return report, cm_df