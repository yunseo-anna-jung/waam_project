from dataset.dataloader import DatasetLoader

loader = DatasetLoader(

    metadata_path="dataset/metadata.csv",

    dataset_root="dataset",

)

x, y = loader.load()

print()

print(x.shape)

print(y.shape)

print(y[:10])