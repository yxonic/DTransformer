import os
from argparse import ArgumentParser

import torch
import tomlkit
from tqdm import tqdm

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer


DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML")
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64)
parser.add_argument("-p", "--with_pid", help="train with pid", action="store_true")
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
    model.to(args.device)
    model.eval()

    # test
    evaluator = Evaluator()

    with torch.no_grad():
        it = tqdm(iter(test_data))
        for batch in it:
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = [None] * len(q)
            for q, s, pid in zip(q, s, pid):
                y, *_ = model.predict(q, s, pid)
                evaluator.evaluate(s, torch.sigmoid(y))
            it.set_postfix(evaluator.report())

    print(evaluator.report())


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
