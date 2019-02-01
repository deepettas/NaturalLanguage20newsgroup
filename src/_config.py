from random import randint


# Classifier Settings:

# Working paths
dataset_path = 'dataset/20-newsgroups/'
inverted_index_path = 'data'
tagged_documents_path = inverted_index_path + '/tagged_docs'

# Number of categories to be randomly chosen
number_of_categories = 20

# Number of documents to compile per category
documents_per_category = 5

# Number of features/characteristics
number_of_features = 5

# Training ratio. Accepted values 0 < x < 1
train_ratio = 0.7

# 1 for Cosine similarity 2 for Jaccard Idex
evaluation_metric = 1

verbose = True
#-----------------------------------------------


class classifierSettings():

    dataset_path = dataset_path
    inverted_index_path = inverted_index_path
    tagged_documents_path = tagged_documents_path

    number_of_categories = number_of_categories
    documents_per_category = documents_per_category
    number_of_features = number_of_features
    train_ratio = train_ratio
    evaluation_metric = evaluation_metric
    verbose = verbose





    def generate_random_settings(self):
        self.number_of_categories = randint(0,20)
        self.documents_per_category = randint(3, 10)
        self.number_of_features = randint(3,7)
        self.train_ratio = randint(1, 9) * 0.1
        self.evaluation_metric = randint(1,2)


