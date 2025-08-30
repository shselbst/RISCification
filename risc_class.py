from entropy import calculate_entropy
from rop import get_num_gadgets
from capstone import *
import random
from elftools.elf.elffile import ELFFile
import sys
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
class RiscClass:
    def __init__(self, bin_file_path):
        self.bin_file_path = bin_file_path
        self.instructions = self.get_inst()


    # returns list of objects containing info about an instruction
    def get_inst(self, arch=CS_ARCH_X86, mode=CS_MODE_64):
    
        elf = ELFFile(open(self.bin_file_path, 'rb'))

        text_start = elf.get_section_by_name('.text')
        text_code = text_start.data()
        text_addr = text_start['sh_addr']
        
        
        md = Cs(arch, mode)
        return list(md.disasm(text_code, text_addr))

    def get_rand_inst_sample(self, out_fname: str, num_inst=50):
        inst_sample = random.sample(self.instructions, num_inst)
        binary_sample = b''.join(ins.bytes for ins in inst_sample)
        with open(out_fname, "wb") as f:
            f.write(binary_sample)
        return inst_sample

    def get_contiguous_rand_inst(self, out_fname: str, num_inst: 50):
        start_index = random.randint(0, len(self.instructions) - num_inst)
        inst_sample = self.instructions[start_index:start_index+num_inst]
        binary_sample = b''.join(ins.bytes for ins in inst_sample)
        with open(out_fname, "wb") as f:
            f.write(binary_sample)
        return inst_sample

    def graph_entropy_cnt_vs_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str) -> None:
        temp_fname = "tmp.bin"
        entropy_values = []
        num_gadgets_values = []

        for i in range(num_repetitions):
            
            self.get_rand_inst_sample(out_fname=temp_fname, num_inst=num_sample_insts)
            entropy = calculate_entropy(temp_fname)
            num_gadgets = get_num_gadgets(temp_fname)
            print(f"Iteration {i + 1}...\nEntropy: {entropy}\n# Gadgets: {num_gadgets}\n")

            entropy_values.append(entropy)
            num_gadgets_values.append(num_gadgets)

        entropy_values = np.array(entropy_values)
        num_gadgets_values = np.array(num_gadgets_values)

        x2 = entropy_values ** 2
        X = np.column_stack((entropy_values, x2))
        X = sm.add_constant(X)

        model = sm.OLS(num_gadgets_values, X).fit()

        print("SUMMARY:", model.summary())


        # Create dot plot
        plt.figure(figsize=(8, 6))
        plt.plot(entropy_values, num_gadgets_values, 'o', markersize=6)
        plt.title('Number of Gadgets vs Entropy')
        plt.xlabel('Entropy')
        plt.ylabel('Number of Gadgets')
        plt.grid(True)
        plt.savefig(graph_fname)  # Save plot to file
        print(f"Plot saved as {graph_fname}")

    def graph_entropy_prcnt_vs_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str) -> None:
        temp_fname = "tmp.bin"
        entropy_values = []
        num_gadgets_values = []

        for i in range(num_repetitions):
            
            self.get_rand_inst_sample(out_fname=temp_fname, num_inst=num_sample_insts)
            entropy = calculate_entropy(temp_fname) / num_sample_insts
            num_gadgets = get_num_gadgets(temp_fname)
            #print(f"Iteration {i + 1}...\nEntropy: {entropy}\n# Gadgets: {num_gadgets}\n")

            entropy_values.append(entropy)
            num_gadgets_values.append(num_gadgets)

        entropy_values = np.array(entropy_values)
        num_gadgets_values = np.array(num_gadgets_values)

        x2 = entropy_values ** 2
        X = np.column_stack((entropy_values, x2))
        X = sm.add_constant(X)

        model = sm.OLS(num_gadgets_values, X).fit()

        print("SUMMARY:", model.summary())


        # Create dot plot
        plt.figure(figsize=(8, 6))
        plt.plot(entropy_values, num_gadgets_values, 'o', markersize=6)
        plt.title('Number of Gadgets vs Entropy %')
        plt.xlabel('Entropy')
        plt.ylabel('Number of Gadgets')
        plt.grid(True)
        plt.savefig(graph_fname)  # Save plot to file
        print(f"Plot saved as {graph_fname}")

    def graph_entropy_prcnt_vs_avg_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str) -> None:
        temp_fname = "tmp.bin"
        entropy_values = []
        num_gadgets_values = []

        for i in range(num_repetitions):
            self.get_rand_inst_sample(out_fname=temp_fname, num_inst=num_sample_insts)
            entropy = calculate_entropy(temp_fname) / num_sample_insts
            num_gadgets = get_num_gadgets(temp_fname)
            #print(f"Iteration {i + 1}...\nEntropy: {entropy}\n# Gadgets: {num_gadgets}\n")
            entropy_values.append(entropy)
            num_gadgets_values.append(num_gadgets)

        # Convert to numpy arrays
        entropy_values = np.array(entropy_values)
        num_gadgets_values = np.array(num_gadgets_values)

        # Convert entropy to percentage (0 to 100)
        entropy_percent = entropy_values * 100

        # Define bins for entropy percentage, e.g., each 1%
        bins = np.linspace(0, 100, num=101)  # edges for 0%,1%,2%,...,100%
        
        # Use pandas to bin and group
        df = pd.DataFrame({
            'entropy_prcnt': entropy_percent,
            'num_gadgets': num_gadgets_values
        })

        # Bin entropy percentages into intervals
        df['entropy_bin'] = pd.cut(df['entropy_prcnt'], bins=bins, right=False)

        # Group by bins and compute average number of gadgets per bin
        grouped = df.groupby('entropy_bin')['num_gadgets'].mean().reset_index()

        # For x-axis, use the bin midpoint (or left edge)
        def bin_midpoint(bin_interval):
            return (bin_interval.left + bin_interval.right) / 2

        grouped['bin_mid'] = grouped['entropy_bin'].apply(bin_midpoint)

        # Remove bins with no data (nan avg)
        grouped = grouped.dropna()

        # Plotting averages per entropy %
        plt.figure(figsize=(10, 6))
        plt.plot(grouped['bin_mid'], grouped['num_gadgets'], marker='o', linestyle='-', color='b')
        plt.title('Average Number of ROP Gadgets vs. Entropy Percentage')
        plt.xlabel('Entropy Percentage (%)')
        plt.ylabel('Average Number of Gadgets')
        plt.grid(True)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(graph_fname)
        print(f"Plot saved as {graph_fname}")

    def graph_entropy_prcnt_vs_gadget_terminators_prcnt(self, num_repetitions: int, num_sample_insts: int, graph_fname: str) -> None:
        temp_fname = "tmp.bin"
        jumps = [
            "jmp",
            "je", "jz",
            "jne", "jnz",
            "js",
            "jns",
            "jo",
            "jno",
            "jc", "jb", "jnae",
            "jnc", "jae", "jnb",
            "jbe",
            "ja",
            "jl", "jnge",
            "jge", "jnl",
            "jle", "jng",
            "jg", "jnle",
            "jp", "jpe",
            "jnp", "jpo",
            "jmpf",
            "jecxz",
            "jrcxz"
        ]
        gadget_terminators = set(jumps + ["call", "ret"])

        entropy_percentages = []
        gadget_terminators_prcnts = []

        for i in range(num_repetitions):
            # Generate random instruction sample first (populates temp_fname)
            inst_sample = self.get_rand_inst_sample(out_fname=temp_fname, num_inst=num_sample_insts)

            # Compute entropy AFTER sample is saved to temp file
            entropy_val = calculate_entropy(temp_fname)  # assuming this returns float entropy (e.g. bits or percent)

            # If entropy_val is like bytes entropy (0-8): scale to percent of max entropy 8 bits
            entropy_percent = (entropy_val / num_sample_insts) * 100

            prcnt_terminators = (sum(1 for inst in inst_sample if inst.mnemonic in gadget_terminators) / len(inst_sample)) * 100

            entropy_percentages.append(entropy_percent)
            gadget_terminators_prcnts.append(prcnt_terminators)

        df = pd.DataFrame({
            'entropy_percent': entropy_percentages,
            'prcnt_terminators': gadget_terminators_prcnts
        })

        # Bin entropy into intervals (e.g., 1% steps), then average gadget terminators per bin:
        bins = np.linspace(0, 100, 101)
        df['entropy_bin'] = pd.cut(df['entropy_percent'], bins=bins, right=False)

        grouped = df.groupby('entropy_bin')['prcnt_terminators'].mean().reset_index()

        # Calculate bin midpoints for x-axis plotting
        grouped['bin_mid'] = grouped['entropy_bin'].apply(lambda x: (x.left + x.right) / 2)

        # Drop NaNs (bins with no samples)
        grouped = grouped.dropna(subset=['prcnt_terminators'])

        # Plot
        plt.figure(figsize=(8, 6))
        plt.plot(grouped['bin_mid'], grouped['prcnt_terminators'], 'o-', markersize=6)
        plt.title('Average Gadget Terminators % vs Entropy Percentage')
        plt.xlabel('Entropy (%)')
        plt.ylabel('Average Gadget Terminators (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_fname)
        plt.close()
        print(f"Plot saved as {graph_fname}")
    
    def graph_gadgets_vs_entropy_multi_line(
        self,
        required_qualifying_samples: int,
        num_sample_insts: int,
        min_terminators_required: int,
        graph_fname: str,
        terminator_counts_to_plot: list = None,
        max_trials: int = 10000
        ) -> None:
        """
        Collect random instruction samples until we have `required_qualifying_samples`
        samples that contain at least `min_terminators_required` gadget terminators.
        Then produce a single multi-line plot where each line shows the average number
        of gadgets vs entropy percentage for a given terminator count.

        Parameters:
        - required_qualifying_samples: number of qualifying samples to collect
          (samples that have >= min_terminators_required terminators).
        - num_sample_insts: how many instructions per sample (passed to get_rand_inst_sample).
        - min_terminators_required: minimum terminators in a sample to count it toward the collection.
        - graph_fname: filename (with extension, e.g., .png) to save the combined plot.
        - terminator_counts_to_plot: optional list of specific terminator counts to include.
          If None, the method plots all observed terminator counts >= min_terminators_required.
        - max_trials: upper bound on attempts to avoid infinite loops; raise RuntimeError if exceeded.
        """
        temp_fname = "tmp.bin"
        jumps = [
            "jmp",
            "je", "jz",
            "jne", "jnz",
            "js",
            "jns",
            "jo",
            "jno",
            "jc", "jb", "jnae",
            "jnc", "jae", "jnb",
            "jbe",
            "ja",
            "jl", "jnge",
            "jge", "jnl",
            "jle", "jng",
            "jg", "jnle",
            "jp", "jpe",
            "jnp", "jpo",
            "jmpf",
            "jecxz",
            "jrcxz"
        ]
        gadget_terminators = set(jumps + ["call", "ret"])

        qualifying_samples = []
        trials = 0
        while len(qualifying_samples) < required_qualifying_samples and trials < max_trials:
            trials += 1
            inst_sample = self.get_rand_inst_sample(out_fname=temp_fname, num_inst=num_sample_insts)
            entropy_val = calculate_entropy(temp_fname)
            # Scale entropy to per-instruction percent (consistent with your other methods)
            entropy_percent = (entropy_val / num_sample_insts) * 100
            term_count = sum(1 for inst in inst_sample if inst.mnemonic in gadget_terminators)
            if term_count < min_terminators_required:
                continue
            num_gadgets = get_num_gadgets(temp_fname)
            qualifying_samples.append({
                'entropy_percent': entropy_percent,
                'num_gadgets': num_gadgets,
                'terminator_count': term_count
            })

        if trials >= max_trials and len(qualifying_samples) < required_qualifying_samples:
            raise RuntimeError(
                f"Exceeded max_trials ({max_trials}) before collecting required qualifying samples "
                f"({required_qualifying_samples}). Collected {len(qualifying_samples)}."
            )

        df = pd.DataFrame(qualifying_samples)

        # decide which terminator counts to plot
        observed_counts = sorted(df['terminator_count'].unique().tolist())
        if terminator_counts_to_plot is None:
            terminator_counts = [c for c in observed_counts if c >= min_terminators_required]
        else:
            # keep only those requested that were actually observed
            terminator_counts = [c for c in sorted(set(terminator_counts_to_plot)) if c in observed_counts]

        if not terminator_counts:
            print("No terminator counts to plot (none observed or none match requested).")
            return

        # bin settings
        bins = np.linspace(0, 100, 101)  # 0..100 percent in 1% steps

        plt.figure(figsize=(10, 7))

        # Choose a color cycle so lines are distinct
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key().get('color', None)

        for idx, tc in enumerate(terminator_counts):
            sub = df[df['terminator_count'] == tc].copy()
            if sub.empty:
                print(f"No samples for terminator count {tc}, skipping.")
                continue
            sub['entropy_bin'] = pd.cut(sub['entropy_percent'], bins=bins, right=False)
            grouped = sub.groupby('entropy_bin')['num_gadgets'].mean().reset_index()

            # compute bin midpoints for x-axis
            def bin_midpoint(bin_interval):
                return (bin_interval.left + bin_interval.right) / 2
            grouped['bin_mid'] = grouped['entropy_bin'].apply(bin_midpoint)
            grouped = grouped.dropna(subset=['num_gadgets'])
            if grouped.empty:
                print(f"After binning, no non-empty bins for terminator count {tc}, skipping.")
                continue

            color = None
            if colors:
                color = colors[idx % len(colors)]
            plt.plot(grouped['bin_mid'], grouped['num_gadgets'],
                     marker='o', linestyle='-', label=f"Terminators = {tc}", color=color)

        plt.title('Average Number of ROP Gadgets vs Entropy Percentage\n(Separate lines per terminator count)')
        plt.xlabel('Entropy Percentage (%)')
        plt.ylabel('Average Number of Gadgets')
        plt.grid(True)
        plt.legend(title='Terminator Count', loc='best', fontsize='small')
        plt.tight_layout()
        plt.savefig(graph_fname)
        plt.close()
        print(f"Combined plot saved as {graph_fname}")