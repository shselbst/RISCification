from typing import List, Dict, Iterable, Optional, Tuple
import random
import os
import logging

from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from elftools.elf.elffile import ELFFile

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# External functions assumed to exist (from your original codebase)
from entropy import calculate_entropy
from rop import get_num_gadgets

# ------------------------
# Module-level constants
# ------------------------
DEFAULT_ARCH = CS_ARCH_X86
DEFAULT_MODE = CS_MODE_64
DEFAULT_TEMP_FNAME = "tmp.bin"
ENTROPY_PERCENT_SCALE = 100.0  # multiply per-instruction entropy fraction by 100 to get percent
DEFAULT_PLOT_SIZE = (8, 6)
ENTROPY_BIN_COUNT = 101  # produces bins for 0..100 in steps of 1
TOP_TERMINATOR_COUNTS = 5

# ------------------------
# Logging configuration
# ------------------------
logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO, fmt: str = None) -> None:
    """
    Configure basic logging for this module. Call once from your application entry point.
    Example: configure_logging(logging.DEBUG)
    """
    if fmt is None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)


# ------------------------
# Helper utilities
# ------------------------
def read_text_section_from_elf(bin_path: str) -> Tuple[bytes, int]:
    """
    Read the .text section bytes and starting address from an ELF file.
    Raises FileNotFoundError or ValueError if section not found.
    """
    with open(bin_path, "rb") as fh:
        elf = ELFFile(fh)
        text_section = elf.get_section_by_name(".text")
        if text_section is None:
            raise ValueError(f"No .text section found in {bin_path}")
        return text_section.data(), text_section["sh_addr"]


def get_jump_and_terminators() -> set:
    """
    Return a set of mnemonics considered ROP gadget terminators (jumps, call, ret).
    Kept as a function to avoid repeating the list in multiple methods.
    """
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
    return set(jumps + ["call", "ret"])


def save_bytes_to_file(data: bytes, out_fname: str) -> None:
    """Write bytes to out_fname."""
    with open(out_fname, "wb") as fh:
        fh.write(data)


def ensure_sample_size_ok(total_insts: int, requested: int) -> None:
    """Raise ValueError if requested sample size is impossible."""
    if requested <= 0:
        raise ValueError("requested sample size must be > 0")
    if requested > total_insts:
        raise ValueError(f"requested {requested} > available instructions {total_insts}")


# ------------------------
# RiscClass (refactored)
# ------------------------
class RiscClass:
    """
    Encapsulates functionality for sampling instructions from an ELF binary,
    computing entropy/gadget metrics on the sampled bytes, and producing plots.
    """

    def __init__(self, bin_file_path: str, arch: int = DEFAULT_ARCH, mode: int = DEFAULT_MODE):
        """
        Initialize the object and disassemble the .text section using Capstone.
        """
        self.bin_file_path = bin_file_path
        self.arch = arch
        self.mode = mode

        # Disassemble and store instructions
        logger.debug("Disassembling binary: %s", bin_file_path)
        self.instructions = self._disassemble_text_section()
        logger.info("Loaded %d instructions from %s", len(self.instructions), bin_file_path)

    # ------------------------
    # Disassembly & sampling
    # ------------------------
    def _disassemble_text_section(self) -> List:
        """
        Disassemble the .text section and return a list of capstone instruction objects.
        """
        text_bytes, text_addr = read_text_section_from_elf(self.bin_file_path)
        md = Cs(self.arch, self.mode)
        return list(md.disasm(text_bytes, text_addr))

    def get_random_instruction_sample(self, num_inst: int, out_fname: str = DEFAULT_TEMP_FNAME) -> List:
        """
        Sample `num_inst` random (non-contiguous) instructions, write their bytes to out_fname,
        and return the list of capstone instruction objects.
        """
        ensure_sample_size_ok(len(self.instructions), num_inst)
        inst_sample = random.sample(self.instructions, num_inst)
        save_bytes_to_file(b"".join(ins.bytes for ins in inst_sample), out_fname)
        logger.debug("Wrote random sample of %d instructions to %s", num_inst, out_fname)
        return inst_sample

    def get_contiguous_instruction_sample(self, num_inst: int, out_fname: str = DEFAULT_TEMP_FNAME) -> List:
        """
        Get a contiguous slice of `num_inst` instructions starting at a random index,
        write their bytes to out_fname, and return the list of instruction objects.
        """
        ensure_sample_size_ok(len(self.instructions), num_inst)
        start_index = random.randint(0, len(self.instructions) - num_inst)
        inst_sample = self.instructions[start_index:start_index + num_inst]
        save_bytes_to_file(b"".join(ins.bytes for ins in inst_sample), out_fname)
        logger.debug("Wrote contiguous sample of %d instructions (start=%d) to %s", num_inst, start_index, out_fname)
        return inst_sample

    # ------------------------
    # Low-level sample metrics
    # ------------------------
    def _entropy_for_sample_file(self, sample_fname: str, per_instruction: bool = False, num_insts: Optional[int] = None) -> float:
        """
        Call external calculate_entropy on the sample file and optionally scale to per-instruction percent.
        Assumes calculate_entropy returns a numeric entropy value (e.g., bits or sum).
        If per_instruction True, then num_insts must be provided and the return is scaled to percent:
            entropy_percent = (entropy_val / num_insts) * 100
        """
        if per_instruction and (num_insts is None or num_insts <= 0):
            raise ValueError("num_insts must be provided and > 0 when per_instruction=True")
        entropy_val = calculate_entropy(sample_fname)
        logger.debug("Calculated entropy for %s: %s", sample_fname, entropy_val)
        if per_instruction:
            scaled = (entropy_val / num_insts) * ENTROPY_PERCENT_SCALE
            logger.debug("Scaled entropy to percent (per-instruction): %s", scaled)
            return scaled
        return float(entropy_val)

    def _count_gadget_terminators_in_sample(self, inst_sample: Iterable) -> int:
        """
        Count the number of instructions in inst_sample whose mnemonic is a gadget terminator.
        Expects each inst to have a `mnemonic` attribute (capstone instruction).
        """
        terminators = get_jump_and_terminators()
        count = sum(1 for inst in inst_sample if getattr(inst, "mnemonic", None) in terminators)
        logger.debug("Counted %d gadget terminators in sample of %d instructions", count, len(list(inst_sample)))
        return count

    # ------------------------
    # Plotting helpers
    # ------------------------
    @staticmethod
    def _fit_quadratic_regression(x: np.ndarray, y: np.ndarray):
        """
        Fit a quadratic (y ~ 1 + x + x^2) OLS model using statsmodels and return the fitted model.
        """
        x2 = x ** 2
        X = np.column_stack((x, x2))
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        return model

    @staticmethod
    def _save_scatter_plot(x: Iterable, y: Iterable, title: str, xlabel: str, ylabel: str, out_fname: str, figsize: Tuple = DEFAULT_PLOT_SIZE):
        """Create and save a simple scatter plot."""
        plt.figure(figsize=figsize)
        plt.plot(x, y, "o", markersize=6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_fname)
        plt.close()

    # ------------------------
    # Public plotting methods
    # ------------------------
    def graph_entropy_cnt_vs_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str,
                                    sample_type: str = "random", temp_fname: str = DEFAULT_TEMP_FNAME) -> None:
        """
        Perform `num_repetitions` samples and plot Number of Gadgets vs Entropy.
        sample_type: "random" (non-contiguous) or "contiguous"
        """
        entropy_values = []
        num_gadgets_values = []

        sampler = self.get_random_instruction_sample if sample_type == "random" else self.get_contiguous_instruction_sample

        for i in range(num_repetitions):
            sampler(num_inst=num_sample_insts, out_fname=temp_fname)
            entropy = self._entropy_for_sample_file(temp_fname, per_instruction=False)
            num_gadgets = get_num_gadgets(temp_fname)
            logger.info("Iteration %d: Entropy=%s, #Gadgets=%s", i + 1, entropy, num_gadgets)
            entropy_values.append(entropy)
            num_gadgets_values.append(num_gadgets)

        x = np.array(entropy_values)
        y = np.array(num_gadgets_values)
        model = self._fit_quadratic_regression(x, y)
        logger.info("Regression SUMMARY:\n%s", model.summary())

        self._save_scatter_plot(x, y, "Number of Gadgets vs Entropy", "Entropy", "Number of Gadgets", graph_fname)
        logger.info("Plot saved as %s", graph_fname)

    def graph_entropy_prcnt_vs_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str,
                                      temp_fname: str = DEFAULT_TEMP_FNAME) -> None:
        """
        Similar to graph_entropy_cnt_vs_gadgets but entropy is scaled per-instruction into percent.
        """
        entropy_values = []
        num_gadgets_values = []

        for i in range(num_repetitions):
            self.get_random_instruction_sample(num_inst=num_sample_insts, out_fname=temp_fname)
            entropy_prcnt = self._entropy_for_sample_file(temp_fname, per_instruction=True, num_insts=num_sample_insts)
            num_gadgets = get_num_gadgets(temp_fname)
            logger.debug("Iteration %d: Entropy(%%)=%s, #Gadgets=%s", i + 1, entropy_prcnt, num_gadgets)
            entropy_values.append(entropy_prcnt)
            num_gadgets_values.append(num_gadgets)

        x = np.array(entropy_values)
        y = np.array(num_gadgets_values)
        model = self._fit_quadratic_regression(x, y)
        logger.info("Regression SUMMARY:\n%s", model.summary())

        self._save_scatter_plot(x, y, "Number of Gadgets vs Entropy %", "Entropy (%)", "Number of Gadgets", graph_fname)
        logger.info("Plot saved as %s", graph_fname)

    def graph_entropy_prcnt_vs_avg_gadgets(self, num_repetitions: int, num_sample_insts: int, graph_fname: str,
                                           temp_fname: str = DEFAULT_TEMP_FNAME) -> None:
        """
        Create a binned line plot of average number of gadgets vs entropy percent.
        """
        entropy_values = []
        num_gadgets_values = []

        for i in range(num_repetitions):
            self.get_random_instruction_sample(num_inst=num_sample_insts, out_fname=temp_fname)
            entropy_prcnt = self._entropy_for_sample_file(temp_fname, per_instruction=True, num_insts=num_sample_insts)
            num_gadgets = get_num_gadgets(temp_fname)
            logger.debug("Iteration %d: Entropy(%%)=%s, #Gadgets=%s", i + 1, entropy_prcnt, num_gadgets)
            entropy_values.append(entropy_prcnt)
            num_gadgets_values.append(num_gadgets)

        df = pd.DataFrame({
            "entropy_prcnt": np.array(entropy_values) * 1.0,  # percent already
            "num_gadgets": np.array(num_gadgets_values)
        })

        bins = np.linspace(0, ENTROPY_PERCENT_SCALE, ENTROPY_BIN_COUNT)
        df["entropy_bin"] = pd.cut(df["entropy_prcnt"], bins=bins, right=False)
        grouped = df.groupby("entropy_bin")["num_gadgets"].mean().reset_index().dropna()

        # compute bin midpoints for plotting
        grouped["bin_mid"] = grouped["entropy_bin"].apply(lambda b: (b.left + b.right) / 2)

        plt.figure(figsize=(10, 6))
        plt.plot(grouped["bin_mid"], grouped["num_gadgets"], marker="o", linestyle="-", color="b")
        plt.title("Average Number of ROP Gadgets vs. Entropy Percentage")
        plt.xlabel("Entropy Percentage (%)")
        plt.ylabel("Average Number of Gadgets")
        plt.grid(True)
        plt.xlim(0, 100)
        plt.tight_layout()
        plt.savefig(graph_fname)
        plt.close()
        logger.info("Plot saved: %s", graph_fname)

    def graph_entropy_prcnt_vs_gadget_terminators_prcnt(self, num_repetitions: int, num_sample_insts: int, graph_fname: str,
                                                       temp_fname: str = DEFAULT_TEMP_FNAME) -> None:
        """
        For many samples, compute entropy percent and the percent of instructions that are gadget terminators,
        then plot average terminator percentage binned by entropy percent.
        """
        entropy_percentages = []
        gadget_terminators_prcnts = []
        terminators = get_jump_and_terminators()

        for i in range(num_repetitions):
            inst_sample = self.get_random_instruction_sample(num_inst=num_sample_insts, out_fname=temp_fname)
            entropy_val = calculate_entropy(temp_fname)
            entropy_percent = (entropy_val / num_sample_insts) * ENTROPY_PERCENT_SCALE
            prcnt_terminators = (self._count_gadget_terminators_in_sample(inst_sample) / len(inst_sample)) * ENTROPY_PERCENT_SCALE
            logger.debug("Iteration %d: Entropy(%%)=%s, Terminators(%%)=%s", i + 1, entropy_percent, prcnt_terminators)
            entropy_percentages.append(entropy_percent)
            gadget_terminators_prcnts.append(prcnt_terminators)

        df = pd.DataFrame({
            "entropy_percent": entropy_percentages,
            "prcnt_terminators": gadget_terminators_prcnts
        })

        bins = np.linspace(0, ENTROPY_PERCENT_SCALE, ENTROPY_BIN_COUNT)
        df["entropy_bin"] = pd.cut(df["entropy_percent"], bins=bins, right=False)
        grouped = df.groupby("entropy_bin")["prcnt_terminators"].mean().reset_index().dropna()
        grouped["bin_mid"] = grouped["entropy_bin"].apply(lambda b: (b.left + b.right) / 2)

        plt.figure(figsize=DEFAULT_PLOT_SIZE)
        plt.plot(grouped["bin_mid"], grouped["prcnt_terminators"], "o-", markersize=6)
        plt.title("Average Gadget Terminators % vs Entropy Percentage")
        plt.xlabel("Entropy (%)")
        plt.ylabel("Average Gadget Terminators (%)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(graph_fname)
        plt.close()
        logger.info("Plot saved: %s", graph_fname)

    def graph_gadgets_vs_entropy_multi_line(
        self,
        required_qualifying_samples: int,
        num_sample_insts: int,
        min_terminators_required: int,
        graph_fname: str,
        terminator_counts_to_plot: Optional[List[int]] = None,
        max_trials: int = 10000,
        temp_fname: str = DEFAULT_TEMP_FNAME
    ) -> None:
        """
        Collect qualifying samples (samples that have >= min_terminators_required termination instructions)
        until required_qualifying_samples are collected or max_trials exceeded. Then produce a multi-line
        plot: each line corresponds to an observed terminator count and shows average number of gadgets per entropy bin.
        """
        terminators = get_jump_and_terminators()
        qualifying_samples: List[Dict] = []
        trials = 0

        while len(qualifying_samples) < required_qualifying_samples and trials < max_trials:
            trials += 1
            inst_sample = self.get_random_instruction_sample(num_inst=num_sample_insts, out_fname=temp_fname)
            term_count = self._count_gadget_terminators_in_sample(inst_sample)
            if term_count < min_terminators_required:
                logger.debug("Trial %d: term_count=%d < min_required=%d -> skipping", trials, term_count, min_terminators_required)
                continue
            entropy_percent = self._entropy_for_sample_file(temp_fname, per_instruction=True, num_insts=num_sample_insts)
            num_gadgets = get_num_gadgets(temp_fname)
            qualifying_samples.append({
                "entropy_percent": entropy_percent,
                "num_gadgets": num_gadgets,
                "terminator_count": term_count
            })
            logger.info("Collected qualifying sample %d/%d (trial=%d): terminators=%d, entropy(%%)=%s, gadgets=%s",
                        len(qualifying_samples), required_qualifying_samples, trials, term_count, entropy_percent, num_gadgets)

        if len(qualifying_samples) < required_qualifying_samples:
            logger.error(
                "Failed to collect required samples: collected %d of %d within %d trials",
                len(qualifying_samples), required_qualifying_samples, max_trials
            )
            raise RuntimeError(
                f"Failed to collect required samples: collected {len(qualifying_samples)} "
                f"of {required_qualifying_samples} within {max_trials} trials"
            )

        df = pd.DataFrame(qualifying_samples)
        observed_counts = sorted(df["terminator_count"].unique().tolist())

        # select top N terminator counts by number of samples (datapoints)
        counts_series = df["terminator_count"].value_counts()  # sorted descending

        if terminator_counts_to_plot is None:
            # choose the top TOP_TERMINATOR_COUNTS most-common terminator counts observed
            terminator_counts = counts_series.index.tolist()[:TOP_TERMINATOR_COUNTS]
        else:
            # respect user-requested counts but only keep those that were actually observed,
            # ordered by observed frequency (most common first)
            requested = set(terminator_counts_to_plot)
            terminator_counts = [c for c in counts_series.index.tolist() if c in requested][:TOP_TERMINATOR_COUNTS]

        if not terminator_counts:
            logger.warning("No terminator counts available to plot; returning.")
            return

        # log which terminator counts will be plotted and their datapoint counts
        logger.info(
            "Plotting up to top %d terminator counts by datapoints: %s",
            TOP_TERMINATOR_COUNTS,
            ", ".join(f"{c}({counts_series[c]})" for c in terminator_counts)
        )

        # bin settings
        bins = np.linspace(0, ENTROPY_PERCENT_SCALE, ENTROPY_BIN_COUNT)  # 0..100 in 1% steps
        plt.figure(figsize=(10, 7))
        prop_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)
        colors = prop_cycle if prop_cycle is not None else []

        for idx, tc in enumerate(terminator_counts):
            sub = df[df["terminator_count"] == tc].copy()
            if sub.empty:
                logger.debug("No samples for terminator count %d, skipping", tc)
                continue
            sub["entropy_bin"] = pd.cut(sub["entropy_percent"], bins=bins, right=False)
            grouped = sub.groupby("entropy_bin")["num_gadgets"].mean().reset_index().dropna()
            if grouped.empty:
                logger.debug("After binning, no non-empty bins for terminator count %d, skipping", tc)
                continue
            # compute bin midpoints for x-axis
            grouped["bin_mid"] = grouped["entropy_bin"].apply(lambda b: (b.left + b.right) / 2)
            x = grouped["bin_mid"].values
            y = grouped["num_gadgets"].values
            color = colors[idx % len(colors)] if colors else None

            # plot the binned averages
            plt.plot(x, y, marker="o", linestyle="-", label=f"Terminators = {tc}", color=color)

            # add a linear trend line (degree=1). Only add if we have at least 2 points.
            if len(x) >= 2:
                # Fit a 1st-degree polynomial (linear)
                coeffs = np.polyfit(x, y, deg=1)
                trend_y = np.polyval(coeffs, x)
                # plot trend as dashed line with same color (or slightly darker)
                plt.plot(x, trend_y, linestyle="--", color=color, alpha=0.8)

            logger.debug("Plotted terminator count %d (%d bins)", tc, len(grouped))

        plt.title("Average Number of ROP Gadgets vs Entropy Percentage\n(Separate lines per terminator count)")
        plt.xlabel("Entropy Percentage (%)")
        plt.ylabel("Average Number of Gadgets")
        plt.grid(True)
        plt.legend(title="Terminator Count", loc="best", fontsize="small")
        plt.tight_layout()
        plt.savefig(graph_fname)
        plt.close()
        logger.info("Combined plot saved: %s", graph_fname)


configure_logging(logging.INFO)