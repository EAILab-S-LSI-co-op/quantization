import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='Model name')
    parser.add_argument('-mp', '--model_path', help='Model name')
    parser.add_argument('-dp', '--data_path', help='Path to data')
    args = parser.parse_args()
    return args