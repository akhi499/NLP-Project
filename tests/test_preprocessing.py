import pytest
from src.preprocessing import normalize_text, matching_occurrences, mark_target

def test_marathi_unicode_and_occurrences():
    sentence="काठ नदीचा काठ आहे."
    assert normalize_text("  काठ\nनदीचा  ")=="काठ नदीचा"
    assert matching_occurrences(sentence,"काठ")==[(0,3),(10,13)]
    assert "[TGT] काठ [/TGT]" in mark_target(sentence,"काठ",1)

def test_missing_target_rejected():
    with pytest.raises(ValueError): mark_target("मराठी वाक्य.","ससा")

