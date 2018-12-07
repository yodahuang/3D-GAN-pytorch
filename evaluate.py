import click
import numpy as np
import torch
from sklearn import svm
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from dataset import ShapeNet, config
from nets import Discriminator

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_category_data(category_datasets, category_labels, discriminator, is_train):
    representations = []
    labels = []
    for dataset, label in zip(category_datasets, category_labels):
        for real_X in tqdm(dataset.gen(is_train)):
            real_X = real_X.to(DEVICE)
            representations.append(discriminator.forward_eval(real_X).detach().cpu().numpy())
            labels.append(np.ones(real_X.shape[0]) * label)
    representations = np.vstack(representations)
    labels = np.hstack(labels)
    return representations, labels


@click.command()
@click.option('-m', '--model_path', required=True, help='Discriminator model path')
@click.option('-b', '--batch_size', type=int, default=64)
def evaluate_single(model_path, batch_size):
    discriminator = Discriminator().to(DEVICE)
    discriminator.eval()
    discriminator.load_state_dict(torch.load(model_path))

    # The idea of config is just pure stupid
    config.set_batchsize(batch_size)

    all_category_names = ['table', 'plant', 'vase', 'curtain', 'dresser', 'laptop', 'person', 'radio', 'bottle', 'bathtub', 'lamp', 'keyboard', 'sink']
    category_labels = range(len(all_category_names))
    category_datasets = [ShapeNet([c], config, infinity=False) for c in all_category_names]

    training_representations, training_labels = get_category_data(category_datasets, category_labels, discriminator, True)

    # Classifier
    clf = svm.LinearSVC(penalty='l2', C=0.01, class_weight='balanced')
    clf.fit(training_representations, training_labels)

    test_representations, test_labels = get_category_data(category_datasets, category_labels, discriminator, False)

    predictions = clf.predict(test_representations)
    print(accuracy_score(test_labels, predictions))


if __name__ == "__main__":
    evaluate_single()
