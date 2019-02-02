from src.classifier.classifier import DocClassifier
from src._config import ClassifierSettings

if __name__ == '__main__':
    Settings = ClassifierSettings()

    # instantiation of classifier, set formation, training & testing
    document_classifier = DocClassifier(Settings)
    document_classifier.form_doc_sets()
    document_classifier.train()
    document_classifier.test()
