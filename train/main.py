from train.scenario_manager import ScenarioManager, InstructionFactory
import os
import argparse
import json
from pprint import pprint

DATASET_PATH = "dataset"
TRACKER_PATH = os.getenv("TRACKER_PATH")

def list_all_possible_instructions():
    folder_path = "train_config"
    files = os.listdir(folder_path)
    json_files = [f for f in files if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the train_config folder.")
        return
    
    instructions = []
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            jf = json.load(f) 
            instructions.append({"name": jf["name"], "content": jf["description"]})

    print("Available instructions:", pprint(instructions))

def call_instruction(instruction_name: str):
    folder_path = "train_config"
    files = os.listdir(folder_path)
    json_files = [f for f in files if f.endswith('.json')]
    if not json_files:
        print("No JSON files found in the train_config folder.")
        return
    
    instruction = None
    for file_name in json_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as f:
            jf = json.load(f) 
            if jf["name"] == instruction_name:
                instruction = jf
                break

    if instruction is None:
        print(f"Instruction '{instruction_name}' not found.")
        return
    
    parsed = InstructionFactory.parse_instruction(instruction)
    sm = ScenarioManager(parsed)
    assert sm is not None

    result = sm.construct().execute()
    print("Execution result:", result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run different functions.")
    parser.add_argument("--instruction_list", help="list all available instructions", action="store_true")
    parser.add_argument("config_name", help="Name of the config file (e.g., base_train)")
    args = parser.parse_args()
    print(args.config_name)

    if args.instruction_list:
        list_all_possible_instructions()
    elif args.config_name:
        call_instruction(args.config_name)
    else:
        print("Please provide an instruction or use --instruction_list to see available instructions.")

