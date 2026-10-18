from src.transformer import sentence_text

def test_cross_encoder_input_marks_exact_target_and_retains_inflected_context():
    row={"sentence":"नदीच्या काठावर घर आहे.","target_word":"काठ","lemma":"काठ","pos":"NOUN","gloss":"नदीची सीमा"}
    assert "[TGT] काठ [/TGT]" in sentence_text(row)
    row["target_word"]="काठा"
    rendered=sentence_text(row)
    assert "[TGT] काठा [/TGT] वर" in rendered
    assert "Target: काठा" in rendered
