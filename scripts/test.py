from argparse import ArgumentParser

import torch
import tomlkit

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer


# configure the main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML")
# load dataset names from configuration
datasets = tomlkit.load(open("data/datasets.toml"))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)


# testing logic
def main(args):
    test_data = KTData(datasets[args.dataset]["test"])

    # prepare model
    model = DTransformer()

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
