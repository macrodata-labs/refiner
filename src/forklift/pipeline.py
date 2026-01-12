from typing import List

from forklift import ForkLiftStep


class ForkLiftPipeline:
    pipeline_steps: List[ForkLiftStep]

    def __init__(self):
        self.pipeline_steps = []

    def __add_step(self, step: ForkLiftStep) -> "ForkLiftPipeline":
        self.pipeline_steps.append(step)
        return self
