from random import randint
import os
from src.classifier.classifier import DocClassifier
from src._config import classifierSettings


def choose_random_categories(num):
    chosen = []
    while len(chosen) < num and len(chosen) <= 20:
        selection = (randint(1,20) - 1)
        if selection  not in chosen:
            chosen.append(selection)

    return chosen



if __name__ == '__main__':


    # Loading all of our untagged files
    untagged_files = [f for f in os.listdir(os.path.abspath(os.path.join(os.pardir, 'dataset/20-newsgroups')))]


    cat_ids = choose_random_categories(len(untagged_files))

    chosen_categories = [untagged_files[cat_id] for cat_id in cat_ids]

    Settings = classifierSettings()

    # actual entry point of algorithm
    dc = DocClassifier(chosen_categories, Settings)
    dc.train()
    dc.test()
