from src.features import DeepFeatures, CosineSimilarity
from src.utils import create_sample_submission
from src.model import load_model

def run_inference(database_dataset, query_dataset, device, threshold=0.6):
    model = load_model(device=device)
    extractor = DeepFeatures(model, device=device)

    features_database = extractor(database_dataset)
    features_query = extractor(query_dataset)

    similarity = CosineSimilarity()(features_query, features_database)

    pred_idx = similarity.argsort(dim=1)[:, -1]
    pred_scores = similarity[range(similarity.shape[0]), pred_idx]

    labels = database_dataset.metadata['identity'].tolist()
    predictions = [labels[i] for i in pred_idx.tolist()]
    new_individual = 'new_individual'
    predictions = [
        pred if score >= threshold else new_individual
        for pred, score in zip(predictions, pred_scores)
    ]

    create_sample_submission(query_dataset, predictions)