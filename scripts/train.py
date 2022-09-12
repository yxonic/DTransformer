from argparse import ArgumentParser

import torch
import tomlkit

from DTransformer.data import KTDataIter
from DTransformer.eval import Evaluator


# configure the main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML", required=True)
# load dataset names from configuration
datasets = tomlkit.load(open("data/datasets.toml"))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)


# training logic
def main(args):
    # prepare dataset
    train_data = KTDataIter(datasets[args.dataset]["train"], shuffle=True)
    valid_data = KTDataIter(datasets[args.dataset]["valid"])

    # prepare model and optimizer
    model = ...
    optim = ...

    # training
    for epoch in range(args.n_epochs):
        for (q, s) in train_data:
            loss = model.get_loss(q, s)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # validation
        evaluator = Evaluator()

        with torch.no_grad():
            for (q, s) in valid_data:
                _, pred = model(q, s)
                evaluator.evaluate(s, pred)

        evaluator.report()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
