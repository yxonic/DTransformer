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
parser.add_argument("-c", "--config", help="configuration file in TOML", required=True)
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64)
parser.add_argument("-n", "--n_epochs", help="training epochs", default=50)
# load dataset names from configuration
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
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
    dataset = datasets[args.dataset]
    train_data = KTData(
        os.path.join(DATA_DIR, dataset["train"]),
        dataset["inputs"],
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_data = KTData(
        os.path.join(DATA_DIR, dataset["valid"]),
        dataset["inputs"],
        batch_size=args.batch_size,
    )

    # prepare model and optimizer
    model = DTransformer(dataset["n_questions"])
    optim = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.0, 0.999), eps=1e-8)
    model.to(args.device)

    # training
    for epoch in range(args.n_epochs):
        model.train()
        for batch in train_data:
            q, s = batch.get("q", "s")
            for q, s in zip(q, s):
                loss = model.get_loss(q, s)
                optim.zero_grad()
                loss.backward()
                optim.step()
            print(loss.item())

        # validation
        model.eval()
        evaluator = Evaluator()

        with torch.no_grad():
            for (q, s) in valid_data:
                pred = model.predict(q, s)
                evaluator.evaluate(s, pred)

        evaluator.report()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
