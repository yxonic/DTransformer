import os
from argparse import ArgumentParser

import torch
import tomlkit
from tqdm import tqdm

from DTransformer.data import KTData
from DTransformer.eval import Evaluator
from DTransformer.model import DTransformer

DATA_DIR = "data"
MODEL_DIR = "model"

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

# training setup
parser.add_argument("-n", "--n_epochs", help="training epochs", default=50, type=int)
parser.add_argument(
    "-lr", "--learning_rate", help="learning rate", type=float, default=1e-3
)
parser.add_argument("-l2", help="L2 regularization", type=float, default=1e-5)
parser.add_argument(
    "-cl", "--cl_loss", help="use contrastive learning loss", action="store_true"
)

# snapshot setup
parser.add_argument("-o", "--output_dir", help="directory to save model files and logs")
parser.add_argument(
    "-f", "--from_file", help="resume training from existing model file", default=None
)


# training logic
def main(args):
    # prepare dataset
    dataset = datasets[args.dataset]
    seq_len = dataset["seq_len"] if "seq_len" in dataset else None
    train_data = KTData(
        os.path.join(DATA_DIR, dataset["train"]),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
        shuffle=True,
    )
    valid_data = KTData(
        os.path.join(
            DATA_DIR, dataset["valid"] if "valid" in dataset else dataset["test"]
        ),
        dataset["inputs"],
        seq_len=seq_len,
        batch_size=args.batch_size,
    )

    # prepare logger and output directory
    # TODO: logger
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        # TODO: no persistency warning
        pass

    # prepare model and optimizer
    model = DTransformer(
        dataset["n_questions"], dataset["n_pid"], shortcut=args.shortcut
    )
    if args.from_file:
        model.load_state_dict(torch.load(args.from_file, map_location=lambda s, _: s))
    optim = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.l2
    )
    model.to(args.device)

    # training
    for epoch in range(1, args.n_epochs + 1):
        print("start epoch", epoch)
        model.train()
        it = tqdm(iter(train_data))
        total_loss = 0.0
        total_cnt = 0
        for batch in it:
            if args.with_pid:
                q, s, pid = batch.get("q", "s", "pid")
            else:
                q, s = batch.get("q", "s")
                pid = [None] * len(q)
            if seq_len is None:
                q, s, pid = [q], [s], [pid]
            for q, s, pid in zip(q, s, pid):
                if args.cl_loss:
                    loss = model.get_cl_loss(q, s, pid)
                else:
                    loss = model.get_loss(q, s, pid)

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()

                total_loss += loss.item()
                total_cnt += 1  # (s >= 0).sum().item()
                it.set_postfix({"loss": total_loss / total_cnt})

        # validation
        model.eval()
        evaluator = Evaluator()

        with torch.no_grad():
            it = tqdm(iter(valid_data))
            for batch in it:
                if args.with_pid:
                    q, s, pid = batch.get("q", "s", "pid")
                else:
                    q, s = batch.get("q", "s")
                    pid = [None] * len(q)
                if seq_len is None:
                    q, s, pid = [q], [s], [pid]
                for q, s, pid in zip(q, s, pid):
                    y, *_ = model.predict(q, s, pid)
                    evaluator.evaluate(s, torch.sigmoid(y))
                it.set_postfix(evaluator.report())

        r = evaluator.report()
        print(r)

        if args.output_dir:
            model_path = os.path.join(
                args.output_dir, f"model-{epoch:02d}-{r['auc']:.4f}.pt"
            )
            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
