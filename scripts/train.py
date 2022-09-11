from argparse import ArgumentParser

from DTransformer.data import KTDataIter


# configure main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML")


def main(args):
    print(args)

    # prepare dataset
    data = KTDataIter()

    # training
    for epoch in range(args.n_epochs):
        for batch in data:
            train_on_batch(batch)


def train_on_batch(batch):
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
