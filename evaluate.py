import click
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import ShapeNet, config


def get_category_data(category_datasets, category_labels, is_train):
    representations = []
    labels = []
    for dataset, label in zip(category_datasets, category_labels):
        for real_X in tqdm(dataset.gen(is_train)):
            batch_size = real_X.size(0)
            real_X = real_X.view(batch_size, -1).numpy()
            representations.append(real_X)
            labels.append(np.ones(real_X.shape[0]) * label)
    representations = np.vstack(representations)
    labels = np.hstack(labels)
    return representations, labels


@click.command()
@click.option('-b', '--batch_size', type=int, default=64)
def evaluate_single(batch_size):

    # The idea of config is just pure stupid
    config.set_batchsize(batch_size)

    all_category_names = ['chair', 'sofa', 'table', 'airplane', 'car']
    category_labels = range(len(all_category_names))
    category_datasets = [ShapeNet([c], config, infinity=False) for c in all_category_names]

    training_representations, training_labels = get_category_data(category_datasets, category_labels, True)

    # Classifier
    clf = svm.LinearSVC(penalty='l2', C=0.01, class_weight='balanced', verbose=1)
    clf.fit(training_representations, training_labels)

    test_representations, test_labels = get_category_data(category_datasets, category_labels, False)

    predictions = clf.predict(test_representations)
    print(accuracy_score(test_labels, predictions))


if __name__ == "__main__":
    evaluate_single()
