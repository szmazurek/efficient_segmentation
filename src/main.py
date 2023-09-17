import argparse
from test import test_lightning

from train import train_lightning
from utils.utils import AVAILABLE_LOSSES, AVAILABLE_MODELS

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default=100, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="Set the device to run the program",
)
parser.add_argument(
    "--training_data_path", type=str, help="Set the path to training dataset"
)
parser.add_argument(
    "--validation_data_path",
    type=str,
    help="Set the path to validation dataset",
)
parser.add_argument(
    "--testing_data_path",
    type=str,
    help="Set the path to testing data (for internal testing",
)
parser.add_argument("--lr", default=0.1, type=float, help="Learning rate")
parser.add_argument(
    "--train", action="store_true", default=False, help="Use True for training"
)
parser.add_argument(
    "--test", action="store_true", default=False, help="Use True for testing"
)
parser.add_argument(
    "--img_size",
    type=int,
    default=256,
    help="input patch size of network input",
)

parser.add_argument(
    "--test_results_save_path",
    type=str,
    help="Set the path to save the test results",
)
parser.add_argument(
    "--model_path", type=str, help="Set the path of the model to be tested"
)
parser.add_argument(
    "--num_classes",
    type=int,
    default=2,
    help="Number of classes / number of output layers",
)

parser.add_argument(
    "--model",
    type=str,
    default="Unet",
    help=f"Model architecture to be used for training, one of {AVAILABLE_MODELS}",
)
parser.add_argument(
    "--loss_function",
    type=str,
    default="DiceLoss",
    help=f"Loss function to be used for training, one of {AVAILABLE_LOSSES}",
)


args = parser.parse_args()


if __name__ == "__main__":
    assert (
        args.model in AVAILABLE_MODELS
    ), f"Model must be one of {AVAILABLE_MODELS}"
    assert (
        args.loss_function in AVAILABLE_LOSSES
    ), f"Loss function must be one of {AVAILABLE_LOSSES}"
    if args.train:
        if args.training_data_path is None:
            raise TypeError(
                "Please specify the path to the training data by setting the parameter "
                '--training_data_path="path_to_training_data"'
            )
        else:
            train_lightning(args)

    elif args.test:
        if args.model_path is None:
            raise TypeError(
                'Please specify the path to model by setting the parameter --model_path="path_to_model"'
            )
        else:
            if args.testing_data_path is None:
                raise TypeError(
                    "Please specify the path to the testing data by setting the parameter "
                    '--testing_data_path="path_to_testing_data"'
                )
            else:
                test_lightning(args)

    else:
        raise TypeError(
            "Please specify the process by setting the parameter "
            '--train or --test to "True"'
        )
