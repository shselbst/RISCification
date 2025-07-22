from capstone import *
import random

# returns list of objects containing info about an instruction
def get_inst(fname: str, arch=CS_ARCH_X86, mode=CS_MODE_64):
    binary = open(fname, 'rb').read()

    md = Cs(CS_ARCH_X86, CS_MODE_64)
    return list(md.disasm(binary, 0x0))

def get_rand_inst_sample(fname: str, out_fname: str, num_inst=50):
    instructions = get_inst(fname)
    print(len(instructions))
    inst_sample = random.sample(instructions, num_inst)
    binary_sample = b''.join(ins.bytes for ins in inst_sample)
    with open(out_fname, "wb") as f:
        f.write(binary_sample)
def get_contiguous_rand_inst(fname: str, out_fname: str, num_inst: 50):
    instructions = get_inst(fname)
    start_index = random.randint(0, len(instructions) - num_inst)
    inst_sample = instructions[start_index:start_index+num_inst]
    binary_sample = b''.join(ins.bytes for ins in inst_sample)
    with open(out_fname, "wb") as f:
        f.write(binary_sample)
