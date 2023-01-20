import linecache
import math
import subprocess
import sys

import torch
from torch.utils.data import DataLoader


class Batch:
    def __init__(self, data, fields, seq_len=None):
        self.data = data
        self.fields = fields
        self.field_index = {f: i for i, f in enumerate(fields)}
        self.seq_len = seq_len

    def get(self, *fields):
        L = len(self.data[0])
        return [
            self.data[self.field_index[f]]
            if self.seq_len is None
            else [
                [
                    self.data[self.field_index[f]][
                        :, i * self.seq_len : (i + 1) * self.seq_len
                    ]
                    for i in range(math.ceil(L / self.seq_len))
                ]
                for f in fields
            ]
            for f in fields
        ]


class KTData:
    def __init__(
        self,
        data_path,
        inputs,
        batch_size=1,
        seq_len=None,
        shuffle=False,
        num_workers=0,
    ):
        self.data = Lines(data_path, group=len(inputs) + 1)
        self.loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=transform_batch,
            num_workers=num_workers,
        )
        self.inputs = inputs
        self.seq_len = seq_len

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return Batch(
            torch.tensor(
                [
                    [int(x) for x in line.strip().split(",")]
                    for line in self.data[index][1:]
                ]
            ),
            self.inputs,
            self.seq_len,
        )


def transform_batch(batch):
    # collect data
    batch_data = [b.data for b in batch]
    # merge configs
    fields, seq_len = batch[0].fields, batch[0].seq_len

    # transpose to separate sequences
    batch = list(zip(*batch_data))
    # pad sequences
    batch = [
        torch.nn.utils.rnn.pad_sequence(
            seqs,
            batch_first=True,
            padding_value=-1,
        )
        for seqs in batch
    ]

    return Batch(batch, fields, seq_len)


class Lines:
    def __init__(self, filename, skip=0, group=1, preserve_newline=False):
        self.filename = filename
        with open(filename):
            pass
        if sys.platform == "win32":
            linecount = sum(1 for _ in open(filename))
        else:
            output = subprocess.check_output(("wc -l " + filename).split())
            linecount = int(output.split()[0])
        self.length = (linecount - skip) // group
        self.skip = skip
        self.group = group
        self.preserve_newline = preserve_newline

    def __len__(self):
        return self.length

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        d = self.skip + 1
        if isinstance(item, int):
            if item < len(self):
                if self.group == 1:
                    line = linecache.getline(self.filename, item + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + item * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                return line

        elif isinstance(item, slice):
            low = 0 if item.start is None else item.start
            low = _clip(low, -len(self), len(self) - 1)
            if low < 0:
                low += len(self)
            high = len(self) if item.stop is None else item.stop
            high = _clip(high, -len(self), len(self))
            if high < 0:
                high += len(self)
            ls = []
            for i in range(low, high):
                if self.group == 1:
                    line = linecache.getline(self.filename, i + d)
                    if not self.preserve_newline:
                        line = line.strip("\r\n")
                else:
                    line = [
                        linecache.getline(self.filename, d + i * self.group + k)
                        for k in range(self.group)
                    ]
                    if not self.preserve_newline:
                        line = [l.strip("\r\n") for l in line]
                ls.append(line)

            return ls

        raise IndexError


def _clip(v, low, high):
    if v < low:
        v = low
    if v > high:
        v = high
    return v
