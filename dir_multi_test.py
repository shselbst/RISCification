from rop import get_num_gadgets
import sys
import matplotlib.pyplot as plt
from risc_class import RiscClass
import numpy as np
import os

def main():
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        raise ValueError("Usage: python3 test.py directory_path num_repetitions num_sample_instructions [max_files]")
    dir_path = sys.argv[1]
    num_repetitions = int(sys.argv[2])
    num_sample_instructions = int(sys.argv[3])
    max_files = None
    if len(sys.argv) == 5:
        max_files = int(sys.argv[4])

    if not os.path.isdir(dir_path):
        raise ValueError(f"Provided path is not a directory: {dir_path}")

    cwd = os.getcwd()  # current working directory for output
    
    files_processed = 0
    for fname in os.listdir(dir_path):
        full_path = os.path.join(dir_path, fname)
        if not os.path.isfile(full_path):
            continue
        print(f"Processing file: {fname}")

        try:
            risc = RiscClass(full_path)
            safe_fname = os.path.splitext(fname)[0]

            # Create output directory in current working dir
            output_dir = os.path.join(cwd, safe_fname)
            os.makedirs(output_dir, exist_ok=True)

            # Save graphs inside this directory
            risc.graph_entropy_cnt_vs_gadgets(
                num_repetitions=num_repetitions,
                num_sample_insts=num_sample_instructions,
                graph_fname=os.path.join(output_dir, f"{safe_fname}_e_cnt_vs_g_cnt.png")
            )
            risc.graph_entropy_prcnt_vs_avg_gadgets(
                num_repetitions=num_repetitions,
                num_sample_insts=num_sample_instructions,
                graph_fname=os.path.join(output_dir, f"{safe_fname}_e_prcnt_vs_g_avg.png")
            )
            risc.graph_entropy_prcnt_vs_gadgets(
                num_repetitions=num_repetitions,
                num_sample_insts=num_sample_instructions,
                graph_fname=os.path.join(output_dir, f"{safe_fname}_e_prcnt_vs_g_cnt.png")
            )
            risc.graph_entropy_prcnt_vs_gadget_terminators_prcnt(
                num_repetitions=num_repetitions,
                num_sample_insts=num_sample_instructions,
                graph_fname=os.path.join(output_dir, f"{safe_fname}_e_prcnt_vs_g_term_prcnt.png")
            )
            files_processed += 1
        except Exception as e:
            print(f"Unable to analyze file {fname}: {e}")
        if max_files is not None and files_processed >= max_files:
            print(f"Reached max_files limit ({max_files}). Stopping.")
            break

if __name__ == "__main__":
    main()