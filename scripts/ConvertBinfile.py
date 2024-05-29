"""
Convert Sitai's first iteration binfiles (double) like

    104006PA.bin
    104006US.bin

to the second iteration bin file format (uint16) like

    104006PAUS.bin

PA aline size: 2730
US aline size: 5460
PAUS aline size: 8192 (2 byte spacer in between)
"""

from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np


PAsize = 2730
USsize = 5460
PAUSsize = 8192

Nalines = 1000


class BinfileFrames:
    """
    Handles reading frames from a binary file.
    """

    def __init__(
        self, binfile: Path, framesize: int, offsetBytes: int, elemSizeBytes: int
    ):
        """
        Params
        ------
        binfile: Path to the binfile
        framesize: size of one frame in bytes
        offsetBytes: number of bytes to skip at the beginning of the file
        elemSizeBytes: number of bytes per element (e.g. 8 for float64)
        """
        self.binfile = binfile
        self.framesizeBytes = Nalines * framesize * elemSizeBytes
        self.nframes = int((binfile.stat().st_size - offsetBytes) / self.framesizeBytes)
        self.offsetBytes = offsetBytes
        self.fp = open(self.binfile, "rb")

    def __len__(self):
        return self.nframes

    def __getitem__(self, idx: int) -> bytes:
        """
        Get one 'frame' in its binary form

        Params
        ------
        idx: index of the frame to retrieve
        """
        self.fp.seek(self.offsetBytes + idx * self.framesizeBytes)
        return self.fp.read(self.framesizeBytes)


def f64_to_u16(x):
    """
    Convert float64 to uint16.
    """
    return ((x + 1) * 2**15).astype(np.uint16)


p = Path(sys.argv[1])
assert p.exists(), f"{p} doesn't exist."


seq = p.name[:6]
pathPA = p.parent / (seq + "PA.bin")
pathUS = p.parent / (seq + "US.bin")
outpath = p.parent / (seq + "PAUS.bin")


itPA = BinfileFrames(pathPA, PAsize, offsetBytes=0, elemSizeBytes=8)
itUS = BinfileFrames(pathUS, USsize, offsetBytes=0, elemSizeBytes=8)
assert len(itPA) == len(itUS)


with open(outpath, "wb") as fp:
    fp.write(b" ")  # 1 byte offset at the start of file
    for i in tqdm(range(len(itPA))):
        framePA = itPA[i]
        frameUS = itUS[i]

        rfPA = f64_to_u16(np.frombuffer(framePA, dtype="<d"))
        rfUS = f64_to_u16(np.frombuffer(frameUS, dtype="<d"))
        rfPA = rfPA.reshape((Nalines, PAsize))
        rfUS = rfUS.reshape((Nalines, USsize))

        for i in range(Nalines):
            ii = i * PAUSsize
            fp.write(rfPA[i])
            fp.write(b"    ")
            fp.write(rfUS[i])
