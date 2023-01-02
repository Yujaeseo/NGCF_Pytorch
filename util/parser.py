import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='NGCF')
    parser.add_argument('-data_path', nargs='?', default='./Data', help='Input data path')
    parser.add_argument('-train_file', nargs='?', default='train.txt', help='Input train file')
    parser.add_argument('-test_file', nargs='?', default='test.txt', help='Input test file')
    parser.add_argument('-dataset', nargs='?', default='gowalla', help='Input dataset')

    parser.add_argument('-lr', type = float, default=0.0001, help='Learning rate')
    parser.add_argument('-embed_size', type=int, default=64, help='Embedding size')
    parser.add_argument('-batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('-node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('-message_dropout', nargs='?', default='[0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('-layer_size', nargs='?', default='[64,64,64]', help='Output sizes of every layer')
    parser.add_argument('-regs', nargs='?', default='[1e-5]', help='Regularizations.')
    parser.add_argument('-epoch', type=int, default=500, help='Epoch')
    parser.add_argument('-ks', nargs='?', default='[20]', help='Output sizes of every layer')
    parser.add_argument('-checkpoint_prefix', nargs='?', default='NGCF_checkpoint', help='Input check point path')

    return parser.parse_args()