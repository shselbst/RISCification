from ropper import RopperService
from typing import Optional
import os
import logging

# Configure a module-level logger. Caller can reconfigure logging as needed.
logger = logging.getLogger(__name__)


def get_gadgets(fname: str,
                all_gadgets: Optional[bool] = False,
                inst_count: Optional[int] = 6,
                arch: Optional[str] = 'x86_64'):
    """
    Return the list of gadgets discovered in `fname` using ropper.RopperService.

    Parameters:
    - fname: path to the file to analyze.
    - all_gadgets: whether to request all gadgets (passed to RopperService options).
    - inst_count: maximum instructions per gadget to consider.
    - arch: architecture string for the target binary (e.g., 'x86_64').

    Notes:
    - This function mirrors the original simple flow: create a RopperService,
      add the file, set architecture, load gadgets, fetch the list, then remove the file.
    - Exceptions from ropper are propagated to the caller.
    """
    # Options for RopperService (kept small and explicit)
    options = {
        'color': False,
        'all': all_gadgets,
        'inst_count': inst_count
    }

    # Create service and perform analysis
    rs = RopperService(options)
    rs.addFile(fname, arch=arch)
    rs.setArchitectureFor(name=fname, arch=arch)
    rs.loadGadgetsFor(name=fname)

    # Retrieve gadgets. Depending on ropper version, .gadgets may be None or a list-like.
    gadgets = rs.getFileFor(name=fname).gadgets

    # Clean up service state: attempt to remove file from RopperService.
    # Log any exception at debug level and continue to return gadgets.
    try:
        rs.removeFile(fname)
    except Exception:
        logger.debug("rs.removeFile(%s) raised an exception; ignoring.", fname, exc_info=True)

    logger.debug("get_gadgets(%s) returned %d gadgets", fname, len(gadgets) if gadgets is not None else 0)
    return gadgets


def get_num_gadgets(fname: str,
                    all_gadgets: Optional[bool] = False,
                    inst_count: Optional[int] = 6,
                    arch: Optional[str] = 'x86_64') -> int:
    """
    Return the number of gadgets found in `fname`.

    This uses get_gadgets internally so the file is only processed once.
    """
    gadgets = get_gadgets(fname, all_gadgets, inst_count, arch)
    count = len(gadgets) if gadgets is not None else 0
    # Log the count at info level for regular visibility
    logger.info("Found %d gadgets in %s", count, fname)
    return count


def get_gadgets_from_list(fname: str,
                          all_gadgets: Optional[bool] = False,
                          inst_count: Optional[int] = 6,
                          arch: Optional[str] = 'x86_64'):
    """
    Convenience function that retrieves gadgets and logs/returns them.

    Kept close to original behavior: previously printed the gadgets; now we log them.
    """
    gadgets = get_gadgets(fname, all_gadgets, inst_count, arch)
    # Use debug logging to avoid noisy output in normal runs; callers can set DEBUG level to see gadgets.
    logger.debug("Gadgets for %s: %s", fname, gadgets)
    return gadgets