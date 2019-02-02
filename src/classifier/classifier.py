import os
import string
from math import ceil, log, sqrt, pow
from random import randint

from src._config import ClassifierSettings
import nltk
from nltk import WordNetLemmatizer

from src.classifier.indexer import DocumentIndexer


def choose_random_categories(num):
    """
    Returns list of num random ids
    for category selection
    :param num:
    :return:
    """
    chosen = []
    while len(chosen) < num:
        selection = (randint(1, 20) - 1)
        if selection not in chosen:
            chosen.append(selection)

    return chosen


class DocClassifier:
    _category_dict = {}
    _characteristics = {}
    _category_models = {}
    _characteristics_strings = []

    def __init__(self, settings=None):

        # Importing the settings
        if settings is None:
            settings = ClassifierSettings().generate_random_settings()

        # Init classifier with selected settings
        self._document_database_path = settings.dataset_path
        self._tagged_docs_path = settings.tagged_documents_path
        self._num_of_categories = settings.number_of_categories
        self._docs_per_cat = settings.documents_per_category
        self._training_ratio = settings.train_ratio
        self._characteristic_num = settings.number_of_features
        self._verbose = settings.verbose
        self._metric_type = settings.evaluation_metric

        # Load all untagged files
        untagged_files = [f for f in os.listdir(os.path.join(os.path.pardir, settings.dataset_path))]

        category_ids = choose_random_categories(self._num_of_categories)

        self._categories = [untagged_files[category_id] for category_id in category_ids]

        for category in self._categories:
            self._category_dict[category] = []

    def form_doc_sets(self):
        """
        Based on the training ratio and _docs_per_cat limit,
        this method groups docs of each category
        into set E (training set) and set A (test set)
        :return:
        """

        if len(self._categories) == 0:
            print('No categories chosen.. Terminating.')
            return

        if self._verbose:
            print('-Forming E and A sets... ', end='')

        for category in self._category_dict.keys():
            # form category dir path and list all docs within
            partial_path = os.path.join(self._document_database_path, category + '/')
            cat_path = os.path.join(os.path.pardir, partial_path)
            doc_list = os.listdir(cat_path)

            # keep only <min(cat_size,docs_per_cat)> random docs of each category
            # ( or all of them if docs_per_cat==-1 )
            if self._docs_per_cat != -1 and self._docs_per_cat < len(doc_list):
                docs_to_remove = len(doc_list) - self._docs_per_cat

                for _ in range(0, docs_to_remove):
                    del doc_list[randint(1, len(doc_list) - 1)]

            # add them to the doc collection
            # and split them to E and A sets
            # based on the training ratio
            num_of_e_docs = int(ceil(len(doc_list) * self._training_ratio))
            for i in range(0, len(doc_list)):
                if i < num_of_e_docs:
                    self._category_dict[category].append((doc_list[i], 'E'))  # (E will have 'E' in the tuple)
                else:
                    self._category_dict[category].append((doc_list[i], 'A'))  # (A will have 'A' in the tuple)

        if self._verbose:
            print('OK')

    def _index_and_extract_characteristics(self):
        """
        Indexing of E set docs and extraction of top
        characteristics (highly weighted terms)
        :return:
        """
        # Gather all docs of E collection
        docs_to_be_indexed = []
        for cat in self._category_dict.keys():
            for doc in self._category_dict[cat]:
                if doc[1] == 'E':
                    docs_to_be_indexed.append(cat + '/' + doc[0])
                else:
                    break

        # Perform indexing
        self._total_docs_in_e = len(docs_to_be_indexed)
        indexer = DocumentIndexer(docs_to_be_indexed, self._verbose)
        indexer.start()

        # 'characteristics' will have a part of the original index,
        # containing only the top characteristics
        self._characteristics = indexer.extract_top_characteristics(self._characteristic_num)

    def _generate_models(self):
        """
        Initialization and update of category model structure.
        :return:
        """
        # model will be a dictionary with category names as keys
        # and for each one a list of tuples
        # (one tuple for every document already belonging in that category).
        # The tuple will contain the filename in first index
        # and the vector for the characteristics in second index
        for category, doc_list in self._category_dict.items():
            self._category_models[category] = []
            for doc in doc_list:
                if doc[1] == 'A':  # we stop at the testing data
                    break
                self._category_models[category].append((doc[0], [0 for _ in range(self._characteristic_num)]))

        # then we will loop our sliced index, and for every lemma/characteristic..
        self._characteristics_strings = list(self._characteristics.keys())

        count = 0
        for characteristic in self._characteristics_strings:
            index_data_for_characteristic = self._characteristics[characteristic]

            # ..we will look at all the documents that contain it..
            for document in index_data_for_characteristic:
                file_name_parts = document['id'].split('/')
                if len(file_name_parts) == 1:
                    file_name_parts = document['id'].split('\\')

                # and for every single document
                # find it in our model structure, in order to update the
                # TF-IDF value of this characteristic
                inner_count = 0
                for doc_model in self._category_models[file_name_parts[0]]:
                    if doc_model[0] == file_name_parts[1]:
                        self._category_models[file_name_parts[0]][inner_count][1][count] = document['w']
                        break
                    inner_count += 1

            count += 1

    def train(self):
        """
        Performs indexing, extraction of characteristics
        and model generation
        :return:
        """

        if self._verbose:
            print('\n-Performing indexing of E set and extraction of characteristics.. ')

        self._index_and_extract_characteristics()

        if self._verbose:
            print('\n-Generating category models.. ', end='')

        self._generate_models()

        if self._verbose:
            print('OK')
            print('\n-Training Complete!')

    @staticmethod
    def _remove_closed_class_categories(tagged):
        ret = []
        ban_list = ['CD', 'CC', 'DT', 'EX', 'IN', 'LS', 'MD', 'PDT', 'POS', 'PRP', 'PRP$', 'RP', 'TO', 'UH', 'WDT',
                    'WP', 'WP$', 'WRB']
        for term in tagged:
            if term[1] not in ban_list:
                ret.append(term)
        return ret

    def _generate_test_model(self, category, filename, wordnet_lemmatizer):
        """
        Calculates the characteristic vector
        of the test doc based on tf-idf metric
        and returns vector
        :param category:
        :param filename:
        :param wordnet_lemmatizer:
        :return:
        """
        test_model = [0 for _ in range(self._characteristic_num)]
        doc_path = os.path.join('../dataset/20_newsgroups/', category, filename)

        with open(doc_path, encoding="Latin-1") as f:
            text_raw = "".join([" " if ch in string.punctuation else ch for ch in f.read()])

            text_raw = text_raw.strip()
            content_tokenized = [wordnet_lemmatizer.lemmatize(w.lower()) for w in nltk.tokenize.word_tokenize(text_raw)]
            content_pos_tagged = nltk.pos_tag(content_tokenized)
            content_pos_tagged_no_closed = self._remove_closed_class_categories(content_pos_tagged)

            # calculate tf for the test model
            for term in content_pos_tagged_no_closed:
                for i in range(len(self._characteristics_strings)):
                    if term[0] == self._characteristics_strings[i]:
                        test_model[i] += 1

            # calculate idf from the E data and update test_model list with the tf*idf value
            for i in range(len(self._characteristics_strings)):
                test_model[i] = test_model[i] * log(
                    self._total_docs_in_e / len(self._characteristics[self._characteristics_strings[i]]))

        return test_model

    def test(self):
        """
        For each category, the test doc is compared via
        cosine or Jaccard similarity metric to the training model
        and results are printed
        :return:
        """
        wordnet_lemmatizer = WordNetLemmatizer()
        total_tests = 0
        correct_decisions = 0
        for category, doc_list in self._category_dict.items():
            # list all docs of the category
            for doc in doc_list:
                if doc[1] == 'E':  # skip training set docs
                    continue

                total_tests += 1
                if self._verbose:
                    print('-Test #' + str(total_tests) + ' category: ' + category + ', doc name: ' + doc[
                        0] + '\n-Generating model.. ', end='')
                model = self._generate_test_model(category, doc[0], wordnet_lemmatizer)
                if self._verbose:
                    print('OK')

                if not self._verbose:
                    print('-Comparing models.. ', end='')
                similarities = {}
                for category_model, doc_models in self._category_models.items():
                    s,c = (0,0)
                    for doc_model in doc_models:
                        c += 1
                        s += self._calc_similarity(doc_model[1], model)
                    similarities[category_model] = s / c
                decision = max(similarities, key=similarities.get)

                if decision == category:
                    correct_decisions += 1
                if self._verbose:
                    print('OK\n-Decision: ' + decision + ' (Accuracy so far: ' + str(
                        correct_decisions * 100 / total_tests) + '%) \n')

        print('Results: ' + str(correct_decisions) + '/' + str(total_tests) + ' (' + str(
            correct_decisions * 100 / total_tests) + '%)')

    def _calc_similarity(self, x, y):
        """
        Performs cosine or jaccard similarity calculation
        based on metric_type setting
        :param x:
        :param y:
        :return:
        """
        if self._metric_type == 1:
            return self._cosine_sim(x, y)
        elif self._metric_type == 2:
            return self._jaccard_index(x, y)

    # noinspection PyBroadException
    def _cosine_sim(self, x, y):
        try:
            a,b,c = (0,0,0) # initializing the variables
            for i in range(0, len(x)):
                a += x[i] * y[i]
                b += pow(x[i], 2)
                c += pow(y[i], 2)
            return a / (sqrt(b) * sqrt(c))
        except Exception:
            # if one of the vectors is all 0 return 0
            return 0

    def _jaccard_index(self, x, y):
        first_set = set(x)
        second_set = set(y)
        index = 1.0
        if first_set or second_set:
            index = (float(len(first_set.intersection(second_set)))
                     / len(first_set.union(second_set)))
        return index
