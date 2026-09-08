from dataset.inspector import DatasetInspector

inspector = DatasetInspector(

    dataset_root="dataset",

    metadata_path="dataset/metadata.csv",

)

result = inspector.summary()