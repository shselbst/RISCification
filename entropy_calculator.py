import sys
import pandas as pd
from capstone import *

def calculate_entropy_with_pandas(filename):
    df = pd.read_csv(filename, header=None, names=['instruction'], skip_blank_lines=True)
    df['instruction'] = df['instruction'].str.strip()
    df = df[df['instruction'] != '']

    return df['instruction'].nunique()

# Takes in binary executable
def calculate_entropy_with_capstone(bin_fname, arch = CS_ARCH_X86, mode = CS_MODE_64):
    md = Cs(arch, mode)
    with open(bin_fname, 'rb') as f:
        bin_code = f.read()

    unique_instructions = set(
        f"{ins.mnemonic} {ins.op_str}".strip() 
        for ins in md.disasm(bin_code, 0x0)
    )

    return len(unique_instructions)

if __name__ == '__main__':
    filename = sys.argv[1]
    entropy = calculate_entropy_with_pandas(filename)
    print(f"Entropy (number of unique instructions): {entropy}")
