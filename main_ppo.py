import vessl

from ppo.train import train
from ppo.cfg import get_cfg


if __name__ == "__main__":
    args = get_cfg()

    if bool(args.vessl):
        vessl.init(organization="snu-eng-dgx", project="PMSP", hp=args)

    train(args)