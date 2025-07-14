from entropy_calculator import calculate_entropy_with_pandas, calculate_entropy_with_capstone
from inst import get_inst, get_rand_inst_sample, get_contiguous_rand_inst
from rop import get_num_gadgets
import sys

def main():
    if len(sys.argv) != 2:
        raise ValueError("Usage: python3 test.py binary_file")

    temp_fname = "tmp.bin"
    entropy = calculate_entropy_with_capstone(sys.argv[0])
    get_rand_inst_sample(fname=sys.argv[0], out_fname=temp_fname, num_inst=100)
    num_gadgets = get_num_gadgets(temp_fname)
    print(f"Random sample yielded...\nEntropy: {str(entropy)}\n# Gadgets: {str(num_gadgets)}")
    
    

if __name__ == "__main__":
    main()
