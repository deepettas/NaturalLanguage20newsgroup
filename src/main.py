from random import randint
import os
from src.classifier.classifier import DocClassifier
from src._config import classifierSettings

if __name__ == '__main__':
    Settings = classifierSettings()

    # actual entry point of algorithm
    document_classifier = DocClassifier(Settings)
    document_classifier.train()
    document_classifier.test()
