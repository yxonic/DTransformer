import os
from argparse import ArgumentParser

import torch
import tomlkit

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer


DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML")
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64)
# load dataset names from configuration
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)


# testing logic
def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        batch_size=args.batch_size,
    )

    # prepare model
    model = DTransformer()
    model.eval()

    # test
    evaluator = Evaluator()

    with torch.no_grad():
        for q, s in test_data:
            pred = model.predict(q, s)
            evaluator.evaluate(s, pred)

    evaluator.report()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
