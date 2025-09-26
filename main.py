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

# # Train baseline model for POS tagging and save results
# baseline_model = pos.train_majority_baseline_pos(train_data)
# # df = pd.DataFrame(baseline_model.items(), columns=['word', 'most_frequent_pos'])
# # df.to_csv("majority_baseline_model.csv", index=False)

# # Evaluate baseline model for POS tagging
# baseline_report = pos.evaluate_baseline_pos(baseline_model, test_data)
# print("Majority baseline evaluation for POS:")
# print(baseline_report)

# # Load spaCy model for POS tagging and evaluate
# nlp = spacy.load("zh_core_web_sm")
# spacy_report, cm_df = pos.evaluate_spacy_pos(nlp, test_data)
# print("SpaCy model evaluation for POS:")
# print(spacy_report)
# print("Confusion matrix:")
# print(cm_df)
# cm_df.to_csv("spacy_pos_confusion_matrix.csv")

# # Save reports to text files
# with open("majority_baseline_report_pos.txt", "w") as f:
#     f.write(baseline_report)
# with open("spacy_report_pos.txt", "w") as f:
#     f.write(spacy_report)

# Evaluate baseline model for dependency parsing
print("\nTraining majority baseline for dependencies...\n")
baseline_model = dependency.train_majority_baseline_dep(train_data)
print(baseline_model)
print("\nMajority baseline evaluation for dependencies:\n")
y_test, y_pred = dependency.evaluate_baseline_dep(baseline_model, test_data)
baseline_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(baseline_report).transpose()
df.to_latex("baseline_classification_report.tex")
df.to_csv("baseline_classification_report.csv") 
print(baseline_report)

# Evaluate spaCy model for dependency parsing
print("\nTraining spaCy model for dependencies...\n")
nlp = spacy.load("zh_core_web_trf")
y_test, y_pred = dependency.evaluate_spacy_dep(nlp, test_data)
print("\nSpaCy model evaluation for dependencies:\n")
spacy_report = classification_report(y_test, y_pred, output_dict=True)
df = pd.DataFrame(baseline_report).transpose()
df.to_latex("spacy_classification_report.tex")
df.to_csv("spacy_classification_report.csv") 
print(spacy_report)
