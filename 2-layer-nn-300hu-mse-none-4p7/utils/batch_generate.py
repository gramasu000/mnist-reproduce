from random import shuffle

from .log import LOG

class BatchIndicesGenerator():
    
    def _random_shuffle(self):
        shuffle(self.list_indices)
    
    def __init__(self, batch_size, num_examples):
        self.num_examples = num_examples
        self.list_indices = list(range(num_examples))
        self.batch_size = batch_size
        self.batch_num = 0
        self._random_shuffle()
        LOG.debug(f"Initialized BatchIndicesGenerator object")

    def gen_batch(self):
        start = self.batch_num * self.batch_size
        end = (self.batch_num + 1) * self.batch_size
        batch_indices = self.list_indices[start:end]
        self.batch_num += 1
        if end >= self.num_examples:
            self.batch_num = 0
            self._random_shuffle() 
        LOG.debug(f"Generated BatchIndices {batch_indices}")
        return batch_indices
