import pytest
from src.data_loader import load_dataset, audit_dataset, split_by_sentence
from pathlib import Path
FIXTURE=Path(__file__).parent/"fixtures"/"sample.csv"

def test_schema_and_audit():
    loaded=load_dataset(FIXTURE); assert len(loaded)==12; assert audit_dataset(loaded)["unique_senses"]==2

def test_schema_rejection():
    with pytest.raises(ValueError): load_dataset(Path(__file__).parent/"fixtures"/"bad.csv")

def test_group_split_no_leakage():
    df=load_dataset(FIXTURE); train,val,test,_=split_by_sentence(df,42,.2,.2)
    assert not (set(train.sentence)&set(val.sentence)); assert not (set(train.sentence)&set(test.sentence)); assert not (set(val.sentence)&set(test.sentence))

