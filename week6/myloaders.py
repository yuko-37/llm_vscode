from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from datasets import load_dataset, Dataset, DatasetDict
from myitems import Item
from typing import List
from tqdm import tqdm


BATCH_SIZE = 1000

class ItemLoader:
    
    def __init__(self, name):
        self.name = name
        self.dataset = None

    def load_datapoint(self, datapoint):
        try:
            price = float(datapoint['price'])
            if price > 0:
                item = Item(datapoint, price, self.name)
                return item if item.include else None
        except ValueError as e:
            return None

    def load_batch(self, batch):
        res = []
        for datapoint in batch:
            result = self.load_datapoint(datapoint)
            # result.category = name
            if result:
                res.append(result)
        return res

    def get_batch(self):
        size = len(self.dataset)
        for i in range(0, size, BATCH_SIZE):
            yield self.dataset.select(range(i, min(i + BATCH_SIZE, size)))


    def load_in_parallel(self, num_workers=8):
        results = []
        total_batches = (len(self.dataset) // BATCH_SIZE) + 1
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for batch in tqdm(pool.map(self.load_batch, self.get_batch()), total=total_batches):
                results.extend(batch)
        return results
    

    def load(self):
        start = datetime.now()
        self.dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", f"raw_meta_{self.name}", split="full")

        # results = []
        # for batch in self.get_batch():
        #     results.extend(self.load_batch(batch))
        
        results = self.load_in_parallel()
        duration = datetime.now() - start
        print((f"Completed {self.name} with {len(results):,} datapoints in {duration.total_seconds()/60:.1f} mins "
              f"({duration.total_seconds():.1f} seconds)"), flush=True)

        return results
