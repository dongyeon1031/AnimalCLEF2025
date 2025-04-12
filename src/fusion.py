from wildlife_tools.similarity.wildfusion import WildFusion

'''
WildFusion 객체 생성 + calibration 수행
'''
def build_wildfusion(matcher_aliked, matcher_mega, calibration_query, calibration_db):
    fusion = WildFusion(
        calibrated_pipelines=[matcher_aliked, matcher_mega],
        priority_pipeline=matcher_mega
    )
    fusion.fit_calibration(calibration_query, calibration_db)
    return fusion