import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='Model name')
    args = parser.parse_args()
    return args

def cnn_qat_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='Model name')
    parser.add_argument('-mp', '--model_path', help='Model path')
    parser.add_argument('-dp', '--data_path', help='Path to data')
    args = parser.parse_args()
    return args

def train_cnn_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_blocks', type=int, help='Model name')
    parser.add_argument('-mp', '--model_path', help='Model name')
    parser.add_argument('-dp', '--data_path', help='Path to data')
    args = parser.parse_args()
    return args

def train_vgg_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='Model name')
    parser.add_argument('-mp', '--model_path', help='Model path')
    parser.add_argument('-dp', '--data_path', help='Path to data')
    args = parser.parse_args()
    return args

def quantize_vgg_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='Model name')
    parser.add_argument('-mp', '--model_path', help='Model path')
    parser.add_argument('-dp', '--data_path', help='Path to data')
    args = parser.parse_args()
    return args