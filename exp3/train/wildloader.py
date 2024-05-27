from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import torchvision.transforms as transforms
import os

def load_wild_train(batch_size, input_img_size):
    root_dir = os.getenv('SLURM_TMPDIR', '/tmp')
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="iwildcam", root_dir=root_dir+"/data", download=True)
    # Get the training set
    train_data = dataset.get_subset(
        "train",
        transform=transforms.Compose(
            [transforms.Resize(input_img_size), transforms.ToTensor()]
        ),
    )
    train_loader = get_train_loader("standard", train_data, batch_size=batch_size, num_workers=6)
    return train_loader #, unlabeled_loader

def load_wild_test(batch_size, input_img_size):
    root_dir = os.getenv('SLURM_TMPDIR', '/tmp')
    # Load the full dataset, and download it if necessary
    dataset = get_dataset(dataset="iwildcam", root_dir=root_dir+"/data", download=True)
    test_data = dataset.get_subset(
        "test",
        transform=transforms.Compose(
            [transforms.Resize(input_img_size), transforms.ToTensor()]
        ),
    )
    # Prepare the evaluation data loader
    test_loader = get_eval_loader("standard", test_data, batch_size=batch_size, num_workers=6)
    return test_loader

# usage;
# for x, y_true, metadata in test_loader:
#     y_pred = model(x)
#     # Accumulate y_true, y_pred, metadata

# # Evaluate
# dataset.eval(all_y_pred, all_y_true, all_metadata)