from datasets import load_dataset, DatasetDict

dataset = load_dataset("scikit-learn/imdb", split="train")
dataset = dataset.rename_columns({"review": "text"})
dataset = dataset.class_encode_column("sentiment")
dataset = dataset.map(lambda x, i: {"_id": i}, with_indices=True)

train_test_set = dataset.train_test_split(test_size=0.5, shuffle=False)
train_dev_set = train_test_set["train"].train_test_split(test_size=0.1, stratify_by_column="sentiment", seed=0)

dataset = DatasetDict({"train": train_dev_set["train"], "eval": train_dev_set["test"], "test": train_test_set["test"]})
dataset.save_to_disk("data/imdb")
