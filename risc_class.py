

from typing import List, Dict, Iterable, Optional, Tuple
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from elftools.elf.elffile import ELFFile
from scipy.stats import linregress

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import random
import os
import logging

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

def _bootstrap_confidence_interval(x, y, degree=1, n_bootstrap=1000, alpha=0.05):
    """Compute bootstrap confidence intervals for polynomial regression predictions."""
    x_range = np.linspace(x.min(), x.max(), 200)
    preds = []
    for _ in range(n_bootstrap):
        sample_idx = np.random.choice(len(x), len(x), replace=True)
        x_sample, y_sample = x[sample_idx], y[sample_idx]
        coeffs = np.polyfit(x_sample, y_sample, deg=degree)
        poly = np.poly1d(coeffs)
        preds.append(poly(x_range))
    preds = np.array(preds)
    lower = np.percentile(preds, 100 * alpha / 2, axis=0)
    upper = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return x_range, lower, upper


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


    def analyze_gadgets_vs_entropy(
        self,
        data_points_per_count: int,
        num_sample_insts: int,
        top_terminator_counts: int,
        terminator_counts_to_plot: Optional[List[int]] = None,
        max_trials: int = 10000000,
        temp_fname: str = DEFAULT_TEMP_FNAME
    ) -> (pd.DataFrame, pd.DataFrame, List[int]):
        """
        Collects random instruction samples, computes full statistics, and returns
        both raw data and summary stats.
        """
        qualifying_samples: List[Dict] = []
        trials = 0

        logger.info("Starting data collection...")
        while trials < max_trials:
            trials += 1
            inst_sample = self.get_random_instruction_sample(
                num_inst=num_sample_insts, out_fname=temp_fname
            )
            term_count = self._count_gadget_terminators_in_sample(inst_sample)
            if term_count < 1:
                continue

            entropy_percent = self._entropy_for_sample_file(
                temp_fname, per_instruction=True, num_insts=num_sample_insts
            )
            num_gadgets = get_num_gadgets(temp_fname)

            qualifying_samples.append({
                "entropy_percent": entropy_percent,
                "num_gadgets": num_gadgets,
                "terminator_count": term_count
            })

            logger.info("Entropy %: " + str(entropy_percent) + " | # Gadgets: " + str(num_gadgets) )

            if trials % 100 == 0:
                logger.debug(f"Trials so far: {trials}, collected: {len(qualifying_samples)} samples")

            df = pd.DataFrame(qualifying_samples)
            counts_series = df["terminator_count"].value_counts()

            if terminator_counts_to_plot is None:
                target_counts = counts_series.index.tolist()[:top_terminator_counts]
            else:
                requested = set(terminator_counts_to_plot)
                target_counts = [
                    c for c in counts_series.index.tolist() if c in requested
                ][:top_terminator_counts]

            enough_data = all(
                counts_series.get(tc, 0) >= data_points_per_count for tc in target_counts
            )
            if enough_data:
                logger.info("Collected enough data for all target terminator counts.")
                break

        if not enough_data:
            raise RuntimeError(
                f"Failed to collect {data_points_per_count} samples for each of "
                f"the top {top_terminator_counts} terminator counts."
            )

        df = pd.DataFrame(qualifying_samples)

        logger.info("Performing statistical analysis...")
        all_results = []
        for tc in target_counts:
            sub = df[df["terminator_count"] == tc]
            slope, intercept, r, p, se = linregress(sub["entropy_percent"], sub["num_gadgets"])
            res = {
                "terminator_count": tc,
                "n": len(sub),
                "slope": slope,
                "intercept": intercept,
                "r_squared": r ** 2,
                "p_value": p,
                "std_err": se,
                "mean_entropy": sub["entropy_percent"].mean(),
                "mean_gadgets": sub["num_gadgets"].mean(),
            }
            all_results.append(res)
            logger.info(
                f"Terminator={tc}: n={res['n']}, slope={res['slope']:.3f}, "
                f"R²={res['r_squared']:.3f}, p={res['p_value']:.3g}"
            )

        slope, intercept, r, p, se = linregress(df["entropy_percent"], df["num_gadgets"])
        logger.info(
            f"Overall Trend: slope={slope:.3f}, intercept={intercept:.3f}, "
            f"R²={r**2:.3f}, p={p:.3g}"
        )

        stats_df = pd.DataFrame(all_results)
        return df, stats_df, target_counts


    def plot_gadgets_vs_entropy(
        self,
        df: pd.DataFrame,
        stats_df: pd.DataFrame,
        graph_fname: str,
        regression_degree: int = 1,
        show_points: bool = True,
        top_n: int = 5
    ) -> None:
        """
        Plot regression lines ONLY for the top N terminator counts by number of points.

        Args:
            df: Raw data DataFrame
            stats_df: Statistics DataFrame
            graph_fname: Output filename
            regression_degree: Degree of polynomial regression
            show_points: Whether to plot scatter points
            top_n: How many top terminator counts to include
        """
        plt.figure(figsize=(12, 8))
        sns.set_context("talk")

        # Determine top N terminator counts
        counts = df["terminator_count"].value_counts()
        top_counts = counts.index[:top_n]
        df = df[df["terminator_count"].isin(top_counts)]
        logger.info(f"Plotting top {top_n} terminator counts: {list(top_counts)}")

        colors = sns.color_palette("tab10", n_colors=len(top_counts))

        # Optional scatter points
        if show_points:
            sns.scatterplot(
                data=df,
                x="entropy_percent",
                y="num_gadgets",
                hue="terminator_count",
                alpha=0.5,
                palette=colors,
                edgecolor=None
            )

        # Trend lines for top N only
        for idx, tc in enumerate(sorted(top_counts)):
            sub = df[df["terminator_count"] == tc]
            x, y = sub["entropy_percent"].values, sub["num_gadgets"].values
            if len(x) < regression_degree + 1:
                continue

            coeffs = np.polyfit(x, y, deg=regression_degree)
            poly = np.poly1d(coeffs)
            x_range = np.linspace(x.min(), x.max(), 200)
            y_pred = poly(x_range)
            plt.plot(x_range, y_pred, color=colors[idx], lw=2, label=f"Term {tc}")

        # Overall trend
        coeffs = np.polyfit(df["entropy_percent"], df["num_gadgets"], deg=regression_degree)
        poly = np.poly1d(coeffs)
        x_range = np.linspace(df["entropy_percent"].min(), df["entropy_percent"].max(), 200)
        y_pred = poly(x_range)
        plt.plot(x_range, y_pred, color="black", lw=3, linestyle="--", label="Overall")

        # Confidence interval for overall
        x_ci, lower, upper = _bootstrap_confidence_interval(
            df["entropy_percent"].values, df["num_gadgets"].values, regression_degree
        )
        plt.fill_between(x_ci, lower, upper, color="black", alpha=0.15)

        plt.title("ROP Gadgets vs Entropy Percentage", fontsize=18)
        plt.xlabel("Entropy Percentage (%)")
        plt.ylabel("Number of ROP Gadgets")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Legend", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(graph_fname, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Plot saved: {graph_fname}")


    def graph_gadgets_vs_entropy_multi_line( 
        self, 
        data_points_per_count = 50, 
        top_terminator_counts = 10, 
        num_sample_insts = 500, 
        graph_fname = "gadgets_vs_entropy_multi_line.png",
        regression_degree = 1,
        show_points = False 
    ): 
        df, stats_df, target_counts = self.analyze_gadgets_vs_entropy( 
            data_points_per_count=data_points_per_count, 
            num_sample_insts=num_sample_insts, 
            top_terminator_counts=top_terminator_counts
        ) 
        self.plot_gadgets_vs_entropy(
            df, 
            stats_df, 
            graph_fname=graph_fname,
            regression_degree=regression_degree, 
            show_points=show_points,
            top_n = top_terminator_counts
        )


configure_logging()