import pytest
from src.inference import WSDPredictor
from src.baseline import train_baseline
from pathlib import Path

def test_candidate_ranking_shape_and_missing_word():
    from src.data_loader import load_dataset
    rows=load_dataset(Path(__file__).parent/"fixtures"/"sample.csv")
    model=train_baseline(rows,max_features=5000,max_iter=100)
    predictor=WSDPredictor.__new__(WSDPredictor); predictor.model_type="baseline"; predictor.model=model; predictor.checkpoint_name="in-memory-test"; predictor.max_length=128; predictor.threshold=.5
    result=predictor.score("नदीच्या काठावर घर आहे.","काठ",["नदीची सीमा","कपड्याची किनार"])
    assert len(result["ranked_candidates"])==2
    assert result["ranked_candidates"][0]["compatibility_score"]>=result["ranked_candidates"][1]["compatibility_score"]
    assert result["confidence"]["kind"]=="uncalibrated compatibility score"
    with pytest.raises(ValueError): predictor.score("घर आहे","काठ",["अर्थ"])

