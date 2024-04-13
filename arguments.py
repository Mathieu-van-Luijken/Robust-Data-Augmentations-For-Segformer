from argparse import ArgumentParser

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data", help="Path to the data")
    parser.add_argument("--batch_size", type=int, default=1, help='the batchsize for the dataloader' )
    parser.add_argument("--num_workers", type=int, default=2, help='')
    parser.add_argument("--random_seed", type=bool, default=False, help='False if a seed is to be set')
    parser.add_argument("--seed", type=int, default=2504, help='the seed')
    parser.add_argument("--lr", type=float, default=0.0006, help='The learning rate')
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--augmenter", type=str, default='base', help="The data augmenter")
    return parser