import sys
import pandas as pd
from capstone import *
from elftools.elf.elffile import ELFFile


# Assumes no metadata
def calculate_entropy(bin_fname, arch = CS_ARCH_X86, mode = CS_MODE_64):
    md = Cs(arch, mode)
    with open(bin_fname, 'rb') as f:
        bin_code = f.read()

    unique_instructions = set(
        f"{ins.mnemonic} {ins.op_str}".strip() 
        for ins in md.disasm(bin_code, 0x0)
    )

    return len(unique_instructions)


def calculate_entropy_with_metadata(bin_fname, arch = CS_ARCH_X86, mode = CS_MODE_64):

    elf = ELFFile(open(bin_fname, 'rb'))

    text_start = elf.get_section_by_name('.text')
    text_code = text_start.data()
    text_addr = text_start['sh_addr']
    md = Cs(arch, mode)
    with open(bin_fname, 'rb') as f:
        bin_code = f.read()

    unique_instructions = set(
        f"{ins.mnemonic} {ins.op_str}".strip() 
        for ins in md.disasm(text_code, text_addr)
    )

    return len(unique_instructions)

