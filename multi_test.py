from rop import get_num_gadgets
import sys
import matplotlib.pyplot as plt
from risc_class import RiscClass

import numpy as np


def main():
    if len(sys.argv) != 4:
        raise ValueError("Usage: python3 multi_test.py binary_file num_repetitions num_sample_instructions")

    risc = RiscClass(sys.argv[1])

    num_repetitions = int(sys.argv[2])
    num_sample_instructions = int(sys.argv[3])

    # risc.graph_entropy_cnt_vs_gadgets(num_repetitions=num_repetitions, num_sample_insts=num_sample_instructions, graph_fname="e_cnt_vs_g_cnt.png")
    # risc.graph_entropy_prcnt_vs_avg_gadgets(num_repetitions=num_repetitions, num_sample_insts=num_sample_instructions, graph_fname="e_prcnt_vs_g_avg.png")
    # risc.graph_entropy_prcnt_vs_gadgets(num_repetitions=num_repetitions, num_sample_insts=num_sample_instructions, graph_fname="e_prcnt_vs_g_cnt.png")
    # risc.graph_entropy_prcnt_vs_gadget_terminators_prcnt(num_repetitions=num_repetitions, num_sample_insts=num_sample_instructions, graph_fname="e_prcnt_vs_g_term_prcnt.png")
    risc.graph_gadgets_vs_entropy_multi_line(
        data_points_per_count=500,
        top_terminator_counts=15,
        num_sample_insts=num_sample_instructions,
        graph_fname="gadgets_vs_entropy_multiline.png"
    )

if __name__ == "__main__":
    main()
