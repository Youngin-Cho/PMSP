import argparse


def get_cfg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--vessl", type=int, default=0, help="whether using vessl or not")
    parser.add_argument("--load_model", type=int, default=0, help="load the trained model")

    parser.add_argument("--load_model_path", type=str, default=None, help="file to load trained models from")
    parser.add_argument("--save_model_dir", type=str, default="./output/train/model/", help="folder to save trained models")
    parser.add_argument("--save_log_dir", type=str, default="./output/train/log/", help="folder to save logs")

    parser.add_argument("--num_episodes", type=int, default=20000, help="Number of episodes to train")
    parser.add_argument("--eval_every", type=int, default=100, help="Evaluate every x frames")
    parser.add_argument("--save_every", type=int, default=1000, help="Save a model every x frames")
    parser.add_argument("--num_job", type=int, default=100, help="the Number of jobs")
    parser.add_argument("--num_machine", type=int, default=5, help="the Number of machine")
    parser.add_argument("--weight_tard", type=float, default=0.5, help="Reward weight of tardiness")
    parser.add_argument("--weight_setup", type=float, default=0.5, help="Reward weight of setup")

    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.899, help="Discount factor gamma")
    parser.add_argument("--lmbda", type=float, default=0.886, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping_parameter")
    parser.add_argument("--K_epoch", type=int, default=3, help="Optimization epoch")
    parser.add_argument("--num_steps", type=int, default=32, help="Number of steps to obtain samples")
    parser.add_argument("--T", type=int, default=10, help="Temperature parameter of soft-max")
    parser.add_argument("--T_step", type=float, default=15000, help="decay step for temperature parameter")
    parser.add_argument("--T_min", type=int, default=1, help="Minimum temperature parameter")
    parser.add_argument("--V_coef", type=float, default=0.113, help="coefficient for value loss")
    parser.add_argument("--E_coef", type=float, default=0.025, help="coefficient for entropy loss")
    parser.add_argument("--n_units", type=int, default=64, help="number of nodes in fully-connected layers")

    args = parser.parse_args()

    return args