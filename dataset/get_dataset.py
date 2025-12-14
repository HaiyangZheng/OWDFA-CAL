from copy import deepcopy

from .owdfa_dataset import get_owdfa_datasets
from .data_utils import MergedDataset


def get_dataset(args, train_transform, test_transform):

    datasets = get_owdfa_datasets(train_transform=train_transform, test_transform=test_transform, train_classes=args.train_classes,
                                 dataset_root= args.dataset_root, predictor_path=args.predictor_path, prop_train_labels=args.prop_train_labels, known_classes=args.known_classes)

    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))
    test_dataset = datasets['test']

    return train_dataset, test_dataset
