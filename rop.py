from ropper import RopperService
from typing import Optional
import os


def get_gadgets(fname: str, all_gadgets: Optional[bool] = False, inst_count: Optional[int] = 6, arch: Optional[str] = 'x86_64'):
    options = {'color' : False,
               'all': all_gadgets,
               'inst_count': inst_count}

    rs = RopperService(options)

    rs.addFile(fname, arch=arch)
    rs.setArchitectureFor(name=fname, arch=arch)
    rs.loadGadgetsFor(name=fname)
    gadgets  = rs.getFileFor(name=fname).gadgets
    rs.removeFile(fname)
    return gadgets

def get_num_gadgets(fname: str, all_gadgets: Optional[bool] = False, inst_count: Optional[int] = 6, arch: Optional[str] = 'x86_64') -> int:
    print(get_gadgets(fname, all_gadgets, inst_count, arch))
    return len(get_gadgets(fname, all_gadgets, inst_count, arch))

def get_gadgets_from_list(fname: str, all_gadgets: Optional[bool] = False, inst_count: Optional[int] = 6, arch: Optional[str] = 'x86_64'):
    gadgets = get_gadgets(fname, all_gadgets, inst_count, arch)
    print(gadgets)

