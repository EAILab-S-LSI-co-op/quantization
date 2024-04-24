import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', help='Model name')
    args = parser.parse_args()
    return args