import os
from argparse import ArgumentParser

import torch
import tomlkit
import matplotlib.pyplot as plt

from DTransformer.data import KTData
from DTransformer.model import DTransformer
from DTransformer.visualize import heat_map


DATA_DIR = "data"

# configure the main parser
parser = ArgumentParser()

# general options
parser.add_argument("--device", help="device to run network on", default="cpu")
parser.add_argument(
    "-s", "--seq_id", help="select a sequence index", default=0, type=int
)

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
parser.add_argument("--d_model", help="model hidden size", type=int, default=128)
parser.add_argument("--n_layers", help="number of layers", type=int, default=1)
parser.add_argument("--n_heads", help="number of heads", type=int, default=8)
parser.add_argument(
    "--n_know", help="dimension of knowledge parameter", type=int, default=32
)

# plot setup
parser.add_argument("-f", "--from_file", help="test existing model file", required=True)


def main(args):
    # prepare datasets
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    test_data = KTData(
        os.path.join(DATA_DIR, dataset["test"]),
        dataset["inputs"],
        seq_len=seq_len,
    )

    # prepare model
    model = DTransformer(
        dataset["n_questions"],
        dataset["n_pid"],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_know=args.n_know,
        n_layers=args.n_layers,
    )

    model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    model.to(args.device)
    model.eval()

    # get one sequence
    data = test_data[args.seq_id]
    q, s, pid = data.get("q", "s", "pid")
    q = q.unsqueeze(0)
    s = s.unsqueeze(0)
    if pid is not None:
        pid = pid.unsqueeze(0)
    with torch.no_grad():
        _, _, _, _, (q_score, k_score) = model.predict(q, s, pid)

    # question attention heatmap
    heads = [0, 1, 2, 3]
    seq_len = 12
    fig1, ax = plt.subplots(1, 4, figsize=(12, 2))
    for i in range(len(heads)):
        im = heat_map(ax[i], q_score[0, heads[i], :seq_len, :seq_len])
    plt.colorbar(im, ax=ax, location="right")

    # knowledge attention heatmap on one head
    steps = [10, 20, 30, 40]
    head = heads[0]
    fig2, ax = plt.subplots(4, 1, figsize=(6, 4))
    for i in range(len(heads)):
        xticks = None if i == len(heads) - 1 else []
        im = heat_map(ax[i], k_score[0, head, steps[i], :, : steps[-1]], xticks=xticks)
    plt.colorbar(im, ax=ax, location="right")

    plt.show()
    # fig.savefig('qt.pdf', bbox_inches='tight')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
