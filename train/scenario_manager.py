from enum import Enum
import math
import time
from typing import List

from dataclasses import dataclass
from train.data_io import Disk
from train.data_cleaner import Cleaner
from train.data_transform import Transformer
from train.model import ModelTrainer
from repositories.repo import Facade



class InstructionFactory:
    def __init__(self):
        pass

    @classmethod
    def parse_instruction(cls, instruction: dict):
        instructions = []
        for inst in instruction["instructions"]:
            instructions.append(InstructionStep(
                type=InstructionEnum.from_string(inst["type"]),
                properties=inst.get("properties", {}),
                call=inst.get("call", [])
            ))
        return Instruction(
            name=instruction["name"],
            description=instruction["description"],
            instructions=instructions,
            repository=instruction["repository"]
        )

class InstructionEnum(Enum):
    DATA_IO = "data_io"
    DATA_CLEANER = "data_cleaner"
    DATA_TRANSFORMER = "data_transformer"
    MODEL_TRAINER = "model_trainer"

    @classmethod
    def from_string(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"Unknown InstructionEnum value: {value}")

@dataclass
class InstructionStep:
    type: InstructionEnum
    properties: dict
    # let the respective class handle the parsing of dictionary
    call: List[dict]

@dataclass
class RepositoryConfig:
    type: str
    properties: dict

@dataclass
class Instruction:
    name: str
    description:str
    instructions: List[InstructionStep]
    repository: dict

class ScenarioManager:
    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        self.pipeline: List[any] = []
        self.facade: Facade = None

    def construct(self):
        facade = Facade.parse_instruction(self.instruction.repository)
        facade.new_run(facade.generate_run_id())
        for step in self.instruction.instructions:
            if step.type == InstructionEnum.DATA_IO:
                self.pipeline.append(Disk.parse_instruction(step.properties, step.call, facade))
            elif step.type == InstructionEnum.DATA_CLEANER:
                self.pipeline.append(Cleaner.parse_instruction(step.properties, step.call, facade))
            elif step.type == InstructionEnum.DATA_TRANSFORMER:
                self.pipeline.append(Transformer.parse_instruction(step.properties, step.call, facade))
            elif step.type == InstructionEnum.MODEL_TRAINER:
                self.pipeline.append(ModelTrainer.parse_instruction(step.properties, step.call, facade))
        self.facade = facade
        return self
    def execute(self):
        recurse = None
        start = time.time()
        for component in self.pipeline:
            recurse = component.execute(recurse)
        self.facade.set_total_runtime((time.time() - start) * 1000.0)
        return recurse
    