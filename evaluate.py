import click
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import ShapeNet, config
from nets import Discriminator


@click.command()
@click.option('-m', '--model_path', required=True, help='Discriminator model path')
@click.option('-b', '--batch_size', type=int, default=64)
def evaluate_single(model_path, batch_size):
    discriminator = Discriminator()
    # discriminator.load_state_dict(torch.load(model_path))

    # The idea of config is just pure stupid
    config.set_batchsize(batch_size)

    all_category_names = ['chair', 'sofa', 'table', 'airplane', 'car']
    category_labels = range(len(all_category_names))
    category_datasets = [ShapeNet([c], config, infinity=False) for c in all_category_names]

    representations = []
    labels = []
    for dataset, label in zip(category_datasets, category_labels):
        for real_X in tqdm(dataset.gen(is_train=True)):
            representations.append(discriminator.forward_eval(real_X).detach().cpu().numpy())
            labels.append(np.ones(real_X.shape[0]) * label)
    representations = np.vstack(representations)
    labels = np.vstack(labels)

    # Classifier
    clf = svm.LinearSVC(penalty='l2', C=0.01, class_weight='balanced')
    clf.fit(representations, labels)

    test_representations = []
    test_labels = []
    for dataset, label in tqdm(zip(category_datasets, category_labels)):
        for real_X in tqdm(dataset.gen(is_train=False)):
            test_representations.append(discriminator.forward_eval(real_X).detach().cpu().numpy())
            test_labels.append(np.ones(real_X.shape[0]) * label)
    predictions = clf.predict(test_representations)
    print(accuracy_score(test_labels, predictions))


if __name__ == "__main__":
    evaluate_single()
