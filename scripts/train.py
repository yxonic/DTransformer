from argparse import ArgumentParser

import torch
import tomlkit

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer


# configure the main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML", required=True)
parser.add_argument("-d", "--device", help="device to run network on", default="cpu")
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
    train_data = KTData(datasets[args.dataset]["train"], shuffle=True)
    valid_data = KTData(datasets[args.dataset]["valid"])

    # prepare model and optimizer
    model = DTransformer()
    optim = torch.optim.Adam(model.parameters())
    model.to(args.device)

    # training
    for epoch in range(args.n_epochs):
        for batch in train_data:
            q, s = batch  # TODO: support more data input types
            loss = model.get_loss(q, s)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # validation
        evaluator = Evaluator()

        with torch.no_grad():
            for (q, s) in valid_data:
                pred = model.predict(q, s)
                evaluator.evaluate(s, pred)

        evaluator.report()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
