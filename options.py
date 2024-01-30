import argparse

parser = argparse.ArgumentParser(description='training argument values')

def add_training_parser(parser):
    parser.add_argument("-device", type=str, default="2,3")
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-lr", type=float, default=1e-4)
    parser.add_argument("-check_interval", type=int, default=5)
    parser.add_argument("-k_fold", type=str, default=10)
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-metrics", type=list, default=["Dice", "Jaccard", "Hausdorff"])
    parser.add_argument("-label_type", type=str, default="Country")
    parser.add_argument("-remark", type=str, default="Globe")