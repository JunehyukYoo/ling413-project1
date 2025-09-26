from collections import defaultdict, Counter
from spacy.tokens import Doc


def extract_clausal_dependency(sentence):
    """
    Given a pyconll sentence, extract clause word order pattern
    (SVO, SOV, VO, SV, V, etc.) based on UD annotations.
    """
    verb, subj, obj = None, None, None

    # 1. Identify the verbs
    for word in sentence:
        if word.upos == 'VERB' and word.head == '0':
            verb = word
            break
    if not verb:
        return None  # skip sentences without verbs
    
    # 2. Find subject and object dependents of the verb if they exist
    for word in sentence:
        if word.head == verb.id:
            if 'subj' in word.deprel:
                subj = word
            elif 'obj' in word.deprel:
                obj = word

    # 3. Determine word order pattern
    elements = []
    if subj: elements.append(("S", int(subj.id)))
    if verb: elements.append(("V", int(verb.id)))
    if obj: elements.append(("O", int(obj.id)))
    label = "".join(t for t,_ in sorted(elements, key=lambda x: x[1]))
    return label

def train_majority_baseline_dep(conll_file):
    '''
    Load UD corpus in CoNLL-U format.
    Returns trained model (count of order type).
    '''
    label_counts = Counter()
    for sentence in conll_file:
        label = extract_clausal_dependency(sentence)
        if label:
            label_counts[label] += 1
    return label_counts

def predict_baseline_dep(model, sentence):
    '''
    Predict dependency ordering for one sentence with heuristics.
    '''
    for word in sentence:
        if "把" in word.form: # special case for "ba" (把) construction
            return "SOV"  
        elif "被" in word.form: # special case for "bei" (被) passive
            if "nsubj" in [w.deprel for w in sentence if w.head == word.id]:  # check if passive
                return "OVS"
            return "OV"  
    return model.most_common(1)[0][0]  # otherwise, predict the most common label

def evaluate_baseline_dep(model, conll_file):
    '''
    Evaluate the baseline model on a test set.
    '''
    actual, prediction = [], []
    for sentence in conll_file:
        label = extract_clausal_dependency(sentence)
        if label:
            actual.append(label)
            prediction.append(predict_baseline_dep(model, sentence))
        
    return actual, prediction

def extract_order_spacy(model, sentence):
    """
    Extracts SVO-style order from a sentence using spaCy's dependency parse.
    Returns label string (e.g. 'SVO', 'VO', 'SOV').
    """
    doc = model(sentence)
    verb, subj, obj = None, None, None

    # Find root verb
    for token in doc:
        if token.pos_ == "VERB":
            verb = token
            break
    if not verb:
        return None  # Skip sentences without clear root verb

    # Find subject and object dependents
    for token in verb.children:
        if "subj" in token.dep_:
            subj = token
        elif "obj" in token.dep_:
            obj = token

    elements = []
    if subj: elements.append(("S", subj.i))
    if verb: elements.append(("V", verb.i))
    if obj:  elements.append(("O", obj.i))

    if not elements:
        return None

    # Sort by index order in the sentence
    elements = sorted(elements, key=lambda x: x[1])
    return "".join(e[0] for e in elements)

def evaluate_spacy_dep(model, conll_file):
    '''
    Evaluate the spaCy model on a test set.
    '''
    skipped = 0
    actual, prediction = [], []
    for sentence in conll_file:
        label = extract_clausal_dependency(sentence)
        if not label:
            continue
        
        words = [w.form for w in sentence]
        # Create spaCy doc with UD tokenization
        pred = extract_order_spacy(model, " ".join(words))
        if pred:
            prediction.append(pred)
            actual.append(label)
        else:
            skipped += 1
    
    if skipped > 0:
        print(f"Skipped {skipped} sentences where spaCy produced no valid order.") 
    return actual, prediction