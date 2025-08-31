def extract_and_disassemble(elf_path):
    with open(elf_path, 'rb') as f:
        elf = ELFFile(f)

        # Find the .text section (usually contains code)
        text_section = elf.get_section_by_name('.text')
        if not text_section:
            print("No .text section found!")
            return

        code = text_section.data()
        addr = text_section['sh_addr']

        # Try to figure out the architecture
        arch = elf.get_machine_arch()
        if arch == 'x86':
            md = Cs(CS_ARCH_X86, CS_MODE_32)
        elif arch == 'x64' or arch == 'AMD64':
            md = Cs(CS_ARCH_X86, CS_MODE_64)
        elif arch == 'ARM':
            md = Cs(CS_ARCH_ARM, CS_MODE_ARM)
        elif arch == 'AARCH64':
            md = Cs(CS_ARCH_ARM64, CS_MODE_ARM)
        else:
            print(f"Unsupported arch: {arch}")
            return

        # Disassemble
        for i in md.disasm(code, addr):
            print("0x%x:\t%s\t%s" % (i.address, i.mnemonic, i.op_str))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python elf_disas.py <yourfile.elf>")
        sys.exit(1)
    extract_and_disassemble(sys.argv[1])