import sys
import pandas as pd


def calculate_entropy_with_pandas(filename):
    df = pd.read_csv(filename, header=None, names=['instruction'], skip_blank_lines=True)
    df['instruction'] = df['instruction'].str.strip()
    df = df[df['instruction'] != '']

    return df['instruction'].nunique()


if __name__ == '__main__':
    filename = sys.argv[1]
    entropy = calculate_entropy_with_pandas(filename)
    print(f"Entropy (number of unique instructions): {entropy}")
