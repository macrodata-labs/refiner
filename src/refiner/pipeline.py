from typing import List

from refiner import RefinerStep


class RefinerPipeline:
    pipeline_steps: List[RefinerStep]

    def __init__(self):
        self.pipeline_steps = []

    def __add_step(self, step: RefinerStep) -> "RefinerPipeline":
        self.pipeline_steps.append(step)
        return self
