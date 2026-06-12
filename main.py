import argparse
import runpy
import sys


MODULE_MAP = {
    "train": "train_whole_sync",
    "test": "test_whole_sync",
    "preprocess": "tools.prepare_musices",
    "prepare-data": "tools.prepare_musices",
    "prepare-viai-a": "tools.prepare_viai_a",
    "split-data": "tools.split_musices",
    "train-viai-a": "train_viai_a",
    "test-viai-a": "test_viai_a",
    "train-viai-av": "train_viai_av",
    "test-viai-av": "test_viai_av",
    "freeze-viai-baselines": "tools.freeze_viai_baselines",
    "prepare-uq-metadata": "tools.prepare_uq_metadata",
}


def _run_module(module_name, passthrough_args):
    old_argv = sys.argv[:]
    try:
        sys.argv = [module_name] + passthrough_args
        runpy.run_module(module_name, run_name="__main__")
    finally:
        sys.argv = old_argv


def build_parser():
    parser = argparse.ArgumentParser(
        description="VIAI unified entrypoint for training and data preprocessing."
    )
    parser.add_argument(
        "action",
        nargs="?",
        choices=sorted(MODULE_MAP.keys()),
        default="train",
        help="Which module to run.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to the target module. Example: -- --name VIAI-AV",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    extra_args = args.extra_args
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    target_module = MODULE_MAP[args.action]
    print("[main] running:", target_module)
    if extra_args:
        print("[main] forwarded args:", " ".join(extra_args))
    _run_module(target_module, extra_args)


if __name__ == "__main__":
    main()
