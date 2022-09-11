from argparse import ArgumentParser


# configure main parser
parser = ArgumentParser()
parser.add_argument("-c", "--config", help="configuration file in TOML")


def main(args):
    print(args)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
