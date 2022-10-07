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
parser.add_argument(
    "-s", "--shortcut", help="short-cut attentive readout", action="store_true"
)
parser.add_argument("-m", "--model", help="choose model")

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

        model = DKT(dataset["n_questions"])
    else:
        from DTransformer.model import DTransformer

        model = DTransformer(
            dataset["n_questions"], dataset["n_pid"], shortcut=args.shortcut
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
