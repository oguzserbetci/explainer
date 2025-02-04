import sys
import os
from datasets import DatasetDict
from train_transformers import TransformersConfig
from collections import Counter

Counter()

if __name__ == '__main__':
    config = TransformersConfig.from_json(os.path.abspath(sys.argv[-1]))
    dataset = DatasetDict.load_from_disk(config.data.data_path)
    breakpoint()
    
    for split in dataset.keys():
        print(f"Split {split} with {len(dataset[split])} samples")
        for task,n in config.data.tasks.items():
            print(f"Task {task} defined with {n} labels")
            labels = [sample[task] for sample in dataset[split]]
            print(f'Split {split} has {len(set(labels))} unique values')
            print(f"{task}: {Counter(labels)}")