from pathlib import Path
import pyconll
import spacy
import dependency
import pos
import pandas as pd
from sklearn.metrics import classification_report

# Define file paths and data
TRAIN, TEST = 'zh_gsd-ud-train.conllu', 'zh_gsd-ud-test.conllu'
print(Path.cwd())
train_fp = Path(Path.cwd() / 'corpora' / TRAIN).resolve()
test_fp = Path(Path.cwd() / 'corpora' / TRAIN).resolve()
train_data = pyconll.load.load_from_file(train_fp)
test_data = pyconll.load.load_from_file(test_fp)

# Evaluate baseline model for dependency parsing
print("\nTraining majority baseline for dependencies...\n")
baseline_model = dependency.train_majority_baseline_dep(train_data)
print(baseline_model)
print("\nMajority baseline evaluation for dependencies:\n")
y_test, y_pred = dependency.evaluate_baseline_dep(baseline_model, test_data)
baseline_report = classification_report(y_test, y_pred)
# df_baseline = pd.DataFrame(baseline_report).transpose()
# df_baseline.to_csv("baseline_classification_report.csv") 
print(baseline_report)

# Evaluate spaCy model for dependency parsing
print("\nTraining spaCy model for dependencies...\n")
nlp = spacy.load("zh_core_web_trf")
y_test, y_pred = dependency.evaluate_spacy_dep(nlp, test_data)
print("\nSpaCy model evaluation for dependencies:\n")
spacy_report = classification_report(y_test, y_pred)
# df_spacy = pd.DataFrame(baseline_report).transpose()
# df_spacy.to_csv("spacy_classification_report.csv") 
print(spacy_report)
