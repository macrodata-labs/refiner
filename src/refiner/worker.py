from refiner import Shard
from refiner.ledger import BaseLedger
from refiner.pipeline import RefinerPipeline


class Worker:
    def __init__(self, rank: int, ledger: BaseLedger, pipeline: RefinerPipeline):
        self.rank: int = rank
        self.ledger: BaseLedger = ledger
        self.pipeline: RefinerPipeline = pipeline

    def run(self):
        while True:
            shard: Shard | None = self.ledger.claim()
            if shard is None:
                break
