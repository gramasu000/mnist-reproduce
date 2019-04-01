"""Module containing functions checking integrity of MNIST binary files

Checks that the information obtained from MNIST binary matches with 
that of the mnist_info module
"""

import os
import hashlib

from .mnist_info import MNIST_SHA3_512, MNIST_SIZES, MNIST_MAGIC_NUM,\
                        MNIST_NUM_EXAMPLES, MNIST_FILENAMES 
from ..utils.log import LOG

def _correct_sha3_512(f, use, type):
    """Checks the MNIST binary's SHA3-512 hash

    Args:
        f (_io.BufferedReader): File object opened in binary mode
        use (str): Either "train" or "test"
        type (str): Either "image" or "label"
    
    Returns:
        bool: True if SHA3 hash of binary file matches 
                record in mnist_info module, False otherwise
    """ 
    correct_digest = MNIST_SHA3_512["{}_{}".format(use, type)]
    m = hashlib.sha3_512()
    m.update(f.read())
    if m.hexdigest() is correct_digest:
        LOG.debug(f"{f.name} SHA3-512 corresponds to MNIST ({use}, {type})")
    else:
        LOG.debug(f"{f.name} SHA3-512 does not correspond to MNIST ({use}, {type})") 
    return m.hexdigest() is correct_digest


def _correct_size(f, use, type):
    """Checks the MNIST binary's file size

    Args:
        f (_io.BufferedReader): File object opened in binary mode
        use (str): Either "train" or "test"
        type (str): Either "image" or "label"
    
    Returns:
        bool: True if file size of binary file matches
                record in mnist_info module, False otherwise    
    """
    correct_fsize = MNIST_SIZES[f"{use}_{type}"]
    f.seek(0, os.SEEK_END)
    fsize = f.tell()
    f.seek(0, os.SEEK_SET)
    if fsize is correct_fsize:
        LOG.debug(f"{f.name} file size corresponds to MNIST ({use}, {type})")
    else:
        LOG.debug(f"{f.name} file size does not correspond to MNIST ({use}, {type})")
    return fsize is correct_fsize


def _correct_magic_num(f, type):
    """Checks the MNIST binary's magic number

    Args:
        f (_io.BufferedReader): File object opened in binary mode
        type (str): Either "image" or "label"
        
    Returns:
        bool: True if magic number of binary file matches
            record in mnist_info module, False otherwise
    """
    correct_fmagicnum = MNIST_MAGIC_NUM[type]
    fmagicnum = int.from_bytes(f.read(4), "big")
    f.seek(0, os.SEEK_SET)
    if fmagicnum is correct_fmagicnum:
        LOG.debug(f"{f.name} magic number corresponds to MNIST {type}")
    else:
        LOG.debug(f"{f.name} magic number does not correspond to MNIST {type}")
    return fmagicnum is correct_fmagicnum


def _correct_num_examples(f, use):
    """Checks the number of dataset examples in MNIST binary
    
    Args:
        f (_io.BufferedReader): File object opened in binary mode
        use (str): Either "train" or "test"
    
    Returns:
        bool: True if number of dataset examples in binary file 
            matches record in mnist_info module, False otherwise
    """ 
    correct_fnumexamples = MNIST_NUM_EXAMPLES[use]
    f.seek(4, os.SEEK_SET)
    fnumexamples = int.from_bytes(f.read(4), "big")
    f.seek(0, os.SEEK_SET)
    if fnumexamples is correct_fnumexamples:
        LOG.debug(f"{f.name} number of dataset examples corresponds to MNIST {use}")
    else:
        LOG.debug(f"{f.name} number of dataset examples does not correspond to MNIST {use}")
    return fnumexamples is correct_fnumexamples


def _is_mnist(f, use, type):
    """Checks the integrity of a particular MNIST binary

    Checks SHA3-512 hash, file size, magic number and 
    number of dataset examples for particular MNIST binary

    Args:
        f (_io.BufferedReader): File object opened in binary mode
        use (str): Either "train" or "test"
        type (str): Either "image" or "label"
       
    Returns:
        bool: True if binary file's information
            corresponds to mnist_info specification. 
    """
    LOG.debug(f"{f.name} - Checking whether this corresponds to MNIST ({use}, {type})")
    mnist_match = _correct_sha3_512(f, use, type) 
                    and _correct_size(f, use, type) 
                    and _correct_magic_num(f, type) 
                    and _correct_num_examples(f, use)
    if mnist_match:
        LOG.debug(f"{f.name} matches MNIST ({use}, {type})")
    else:
        LOG.debug(f"{f.name} does not match MNIST ({use}, {type})") 
    return mnist_match


def _check_file(f):
    """Checks whether binary file corresponds with an MNIST dataset
    
    Checks whether _is_mnist(f, use, type) is true
    for any valid value of use and type.
    use is either "train" or "test" and type is either "image" or "label".
    We return (use, type) tuple.
    If _is_mnist is always False, we return (None, None) 
    
    Args:
        f (_io.BufferedReader): File object opened in binary mode
    
    Returns:
        str, str : use, type
    """
     
    for use in ["train", "test"]:
        for type in ["image", "label"]:
            if _is_mnist(f, use, type):
                LOG.info(f"{f.name} matches MNIST ({use}, {type})")
                return use, type
    LOG.warning(f"{f.name} not recognized as MNIST data")
    return None, None


def check_mnist():
    """Runs _check_file on four specific binary files in assets/ directory

    Returns:
        True if all four binary files in assets/ are MNIST datasets.
        False otherwise.
    
    Note:
        This should return True.
    """

    for key, filepath in MNIST_FILENAMES:
        with open(filepath, "rb") as f:
            correct_key = "{}_{}".format(*_check_file(f))
            if key is not correct_key
                LOG.warning("MNIST_FILENAMES not accurate - Do not extract.")
                return False
    LOG.info("MNIST_FILENAMES have been verified - MNIST extraction can begin.")
    return True
