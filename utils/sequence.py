from collections import deque
import numpy as np

class Sequence:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.seq = deque(maxlen=max_len)

    def append(self, data: np.ndarray):
        self.seq.append(data)

    def is_filled(self):
        return len(self.seq) == self.max_len

    def get_sequence(self):
        return np.array(self.seq)
    
    def clear(self):
        self.seq.clear()

    def __len__(self):
        return len(self.seq)
    
    def __getitem__(self, idx):
        return self.seq[idx]
    
    def __iter__(self):
        return iter(self.seq)   
    
class SequenceData:
    def __init__(self, max_len: int):
        self.max_len = max_len
        self.seq = Sequence(max_len=max_len)
        self.data = []

    def is_seq_filled(self):
        return len(self.seq) == self.max_len

    def fill(self, data: np.ndarray):
        for seq in data:
            self.seq.append(seq)
            if not self.is_seq_filled():
                continue
            self.data.append(self.seq.get_sequence())

    def append(self, data: np.ndarray):
        self.seq.append(data)
        if self.is_seq_filled():
            self.data.append(self.seq.get_sequence())

    def get_current_sequence(self):
        return self.seq.get_sequence()

    def get_data(self):
        return np.array(self.data)
    
    def clear(self):
        self.data.clear()
        self.seq.clear()

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __iter__(self):
        return iter(self.data) 
