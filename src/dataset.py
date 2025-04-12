from wildlife_datasets.datasets import AnimalCLEF2025

'''
AnimalCLEF2025 로드하고, query/database/calibration 분리까지 담당
'''
def load_datasets(root, calibration_size=100):
    dataset = AnimalCLEF2025(root, load_label=True)

    dataset_database = dataset.get_subset(dataset.metadata['split'] == 'database')
    dataset_query = dataset.get_subset(dataset.metadata['split'] == 'query')

    dataset_calibration = AnimalCLEF2025(root, df=dataset_database.metadata[:calibration_size], load_label=True)

    return dataset, dataset_database, dataset_query, dataset_calibration