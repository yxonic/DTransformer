import os
from argparse import ArgumentParser

import torch
import tomlkit
from tqdm import tqdm

from DTransformer.data import KTData
from DTransformer.eval import Evaluator

DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument("-bs", "--batch_size", help="batch size", default=64, type=int)

# data setup
datasets = tomlkit.load(open(os.path.join(DATA_DIR, "datasets.toml")))
parser.add_argument(
    "-d",
    "--dataset",
    help="choose from a dataset",
    choices=datasets.keys(),
    required=True,
)
parser.add_argument(
    "-p", "--with_pid", help="provide model with pid", action="store_true"
)

# model setup
# TODO: model size, dropout rate, etc.
parser.add_argument("-m", "--model", help="choose model")
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=1)

# test setup
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)


# testing logic
def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    # prepare model
    if args.model == "DKT":
        from baselines.DKT import DKT

        model = DKT(dataset["n_questions"], args.d_model)
    elif args.model == "DKVMN":
        from baselines.DKVMN import DKVMN

        model = DKVMN(dataset["n_questions"], args.batch_size)
    elif args.model == "AKT":
        from DTransformer.model import DTransformer

        model = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            shortcut=True,
            dropout=args.dropout,
        )
    else:
        from DTransformer.model import DTransformer

        model = DTransformer(
            dataset["n_questions"],
            dataset["n_pid"],
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )

    model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
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
                pid = None if seq_len is None else [None] * len(q)
            if seq_len is None:
                q, s, pid = [q], [s], [pid]
            for q, s, pid in zip(q, s, pid):
                q = q.to(args.device)
                s = s.to(args.device)
                if pid is not None:
                    pid = pid.to(args.device)
                y, *_ = model.predict(q, s, pid)
                evaluator.evaluate(s, torch.sigmoid(y))
            # it.set_postfix(evaluator.report())

    print(evaluator.report())


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
