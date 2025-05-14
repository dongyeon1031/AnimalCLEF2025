from wildlife_tools.similarity.wildfusion import WildFusion

def build_wildfusion(calibration_query, calibration_db, *pipelines, priority_pipeline):
    """
    Build a WildFusion object with an arbitrary number of calibrated pipelines.
    Args:
        calibration_query : dataset for calibration (query side)
        calibration_db    : dataset for calibration (db side)
        *pipelines        : any number of SimilarityPipeline objects
        priority_pipeline : pipeline used for tie‑breaks
    """
    fusion = WildFusion(
        calibrated_pipelines=list(pipelines),
        priority_pipeline=priority_pipeline
    )
    fusion.fit_calibration(calibration_query, calibration_db)
    return fusion