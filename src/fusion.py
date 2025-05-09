from wildlife_tools.similarity.wildfusion import WildFusion

def build_wildfusion(calibration_query, calibration_db, *pipelines, priority_pipeline):
    """
    Build WildFusion using an arbitrary number of calibrated pipelines.
    First positional arg after calib datasets should be the list of pipelines.
    priority_pipeline chooses tie‑break order.
    """
    fusion = WildFusion(
        calibrated_pipelines=list(pipelines),
        priority_pipeline=priority_pipeline
    )
    fusion.fit_calibration(calibration_query, calibration_db)
    return fusion