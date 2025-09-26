from collections import defaultdict, Counter
from spacy.tokens import Doc

def extract_clausal_dependencies(sentence):
    """
    Given a pyconll sentence, extract *all* clause-level word order patterns
    (SVO, SOV, VO, SV, V, etc.) based on UD annotations.
    Returns a list of labels (one per verb clause).
    """
    labels = []

    # 1. Loop over all verbs in the sentence
    for verb in [w for w in sentence if w.upos == 'VERB']:
        subj, obj = None, None

        # 2. Find subject and object dependents of this verb
        for word in sentence:
            if word.head == verb.id:
                if 'subj' in word.deprel:
                    subj = word
                elif 'obj' in word.deprel:
                    obj = word
                elif 'obl:agent' in word.deprel:   # 被-agent
                    subj = word
                elif 'obl:patient' in word.deprel: # 把-patient
                    obj = word

        # 3. Build clause elements in surface order
        elements = []
        if subj: elements.append(("S", int(subj.id)))
        elements.append(("V", int(verb.id)))
        if obj:  elements.append(("O", int(obj.id)))

        if elements:
            label = "".join(t for t, _ in sorted(elements, key=lambda x: x[1]))
            labels.append(label)

    return labels if labels else None

def train_majority_baseline_dep(conll_file):
    '''
    Load UD corpus in CoNLL-U format.
    Returns trained model (count of order type).
    '''
    label_counts = Counter()
    for sentence in conll_file:
        labels = extract_clausal_dependencies(sentence)
        if labels:
            for label in labels:
                label_counts[label] += 1
    return label_counts

def predict_baseline_dep(model, sentence):
    """
    Predict dependency orderings for all clauses (verbs) in a sentence.
    Uses simple heuristics (BA/BEI) or falls back on majority baseline.
    Returns a list of predicted labels, one per clause.
    """
    predictions = []

    # Loop over all verbs in the sentence
    verbs = [w for w in sentence if w.upos == "VERB"]
    if not verbs:
        return []  # no predictions if no verbs

    for verb in verbs:
        # Collect dependents for heuristic layer
        subj = None
        for w in sentence:
            if w.head == verb.id:
                if "subj" in w.deprel:
                    subj = w

        # --- Heuristics ---
        # BA construction (把) → SOV
        if any("把" in w.form for w in sentence):
            predictions.append("SOV")
            continue

        # BEI passive (被) → OVS if subj present, else OV
        if any("被" in w.form for w in sentence):
            has_subj = subj is not None
            predictions.append("OVS" if has_subj else "OV")
            continue

        # --- Default order from majority ---
        predictions.append(model.most_common(1)[0][0])  # fallback

    return predictions

def evaluate_baseline_dep(model, conll_file):
    '''
    Evaluate the baseline model on a test set.
    '''

    actual, prediction = [], []
    for sentence in conll_file:
        labels = extract_clausal_dependencies(sentence)  # gold labels
        if not labels:
            continue

        preds = predict_baseline_dep(model, sentence)  # predicted labels

        # If baseline predicted fewer labels than gold, pad with "N/A"
        if len(preds) < len(labels):
            preds += ["N/A"] * (len(labels) - len(preds))

        # If baseline predicted more, truncate
        if len(preds) > len(labels):
            preds = preds[:len(labels)]

        actual.extend(labels)
        prediction.extend(preds)

    return actual, prediction

def extract_order_spacy(model, sentence):
    """
    Extracts SVO-style word order patterns from all clauses in a sentence
    using spaCy's dependency parse.
    Returns a list of labels (e.g. ["SVO", "VO"]).
    """
    doc = model(sentence)
    labels = []

    # Loop over all verbs
    verbs = [t for t in doc if t.pos_ == "VERB"]
    if not verbs:
        return []  # no verbs → no labels

    for verb in verbs:
        subj, obj = None, None

        # Find subject and object dependents of this verb
        for child in verb.children:
            if "subj" in child.dep_:
                subj = child
            elif "obj" in child.dep_:
                obj = child

        # Build clause elements
        elements = []
        if subj: elements.append(("S", subj.i))
        elements.append(("V", verb.i))
        if obj:  elements.append(("O", obj.i))

        if elements:
            label = "".join(t for t, _ in sorted(elements, key=lambda x: x[1]))
            labels.append(label)

    return labels

def evaluate_spacy_dep(model, conll_file):
    '''
    Evaluate the spaCy model on a test set.
    '''
    actual, prediction = [], []

    for sentence in conll_file:
        labels = extract_clausal_dependencies(sentence)  # gold labels
        if not labels:
            continue

        words = [w.form for w in sentence]
        preds = extract_order_spacy(model, " ".join(words))  # predicted labels

        if preds is None:
            preds = []

        # If spaCy predicted fewer labels than gold, pad with "N/A"
        if len(preds) < len(labels):
            preds += ["N/A"] * (len(labels) - len(preds))

        # If spaCy predicted more, truncate
        if len(preds) > len(labels):
            preds = preds[:len(labels)]

        assert len(actual) == len(prediction), f"Mismatch: {len(actual)} vs {len(prediction)}"

        actual.extend(labels)
        prediction.extend(preds)

    return actual, prediction