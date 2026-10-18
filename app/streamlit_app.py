from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference import WSDPredictor
from src.preprocessing import matching_occurrences
from src.wic_inference import SenWiChWiCPredictor

st.set_page_config(page_title="MarathiSense",page_icon="🔎",layout="wide")
st.title("MarathiSense")
st.caption("Context-aware Marathi word sense disambiguation with candidate-ranking models.")
KAGGLE_BASELINE="checkpoints/kaggle_clean_candidate/baseline.joblib"
DEFAULT_BASELINE=KAGGLE_BASELINE if (ROOT/KAGGLE_BASELINE).is_file() else "checkpoints/baseline.joblib"

def checkpoint_path(path):
    candidate=Path(path).expanduser()
    return candidate if candidate.is_absolute() else ROOT/candidate

@st.cache_resource(show_spinner="Loading model…")
def load_predictor(path,kind,max_length,threshold):
    return WSDPredictor(str(checkpoint_path(path)),kind,max_length=max_length,threshold=threshold)

@st.cache_resource(show_spinner="Loading SenWiCh WiC model…")
def load_wic_predictor(path,max_length):
    return SenWiChWiCPredictor(str(checkpoint_path(path)),max_length=max_length)

def parse_candidates(value, model_type, separator="\n"):
    entries=[entry.strip() for entry in value.split(separator) if entry.strip()]
    if model_type!="wsd_outputs": return entries
    candidates=[]
    for entry in entries:
        sense_id,has_gloss,gloss=entry.partition("|")
        sense_id=sense_id.strip()
        if not sense_id: raise ValueError("Enter each sense as `sense_id | optional definition`.")
        candidate={"sense_id":sense_id}
        if has_gloss and gloss.strip(): candidate["gloss"]=gloss.strip()
        candidates.append(candidate)
    return candidates

with st.sidebar:
    st.header("Model")
    model_type=st.selectbox("Model type",["baseline","transformer","wsd_outputs"])
    default_path={"baseline":DEFAULT_BASELINE,"transformer":"checkpoints/muril","wsd_outputs":"wsd_outputs/models/IndicBERT-v2.pt"}[model_type]
    model_path=st.text_input("Checkpoint path",default_path,key=f"checkpoint_{model_type}")
    max_length=st.number_input("Transformer max sequence length",64,512,256,step=32)
    threshold=st.slider("Low-score warning threshold",0.0,1.0,.5,.05)
    st.caption("The wsd_outputs checkpoints use their matching Hugging Face tokenizer and config, downloaded on first use if needed. They rank sentence/lemma/sense-ID pairs, so provide sense IDs for candidates. IndicBERT-v2 is selected by default; MahaBERT-v2, MahaRoBERTa, and XLM-R-large are also available. The MuRIL adapter needs its base model available locally or downloadable from Hugging Face. Scores are raw compatibility scores, not calibrated probabilities.")

def predictor():
    resolved=checkpoint_path(model_path)
    if not resolved.exists(): raise FileNotFoundError(f"Checkpoint not found: {resolved}.")
    return load_predictor(str(resolved),model_type,int(max_length),float(threshold))

single,batch,compare,wic,info=st.tabs(["Single sentence","Batch CSV","Model comparison","SenWiCh WiC","Model information"])
with single:
    sentence=st.text_area("Marathi sentence",placeholder="मराठी वाक्य लिहा…")
    target=st.text_input("Target word")
    candidate_label="Candidate senses (sense_id | optional definition, one per line)" if model_type=="wsd_outputs" else "Candidate definitions (one per line)"
    candidate_hint="Example: 301 | एखाद्या मार्गाचा किंवा वाटेचा अर्थ" if model_type=="wsd_outputs" else "प्रत्येक संभाव्य अर्थाची व्याख्या एका ओळीत"
    candidates_text=st.text_area(candidate_label,placeholder=candidate_hint)
    occ=0
    if sentence and target:
        count=len(matching_occurrences(sentence,target))
        if count:
            occ=st.selectbox("Target occurrence",options=list(range(1,count+1)),format_func=lambda i:f"{i} / {count}")-1
    if st.button("Disambiguate",type="primary"):
        try:
            candidates=parse_candidates(candidates_text,model_type)
            result=predictor().score(sentence,target,candidates,occurrence=occ)
            st.subheader("Selected meaning"); st.write(result["selected_sense"]["definition"])
            spans=matching_occurrences(sentence,target)
            if spans:
                start,end=spans[occ]; highlighted=sentence[:start]+"**"+sentence[start:end]+"**"+sentence[end:]
                st.markdown(highlighted)
            else: st.markdown(sentence)
            st.dataframe(pd.DataFrame(result["ranked_candidates"]),use_container_width=True,hide_index=True)
            st.info(result["warning"] or "Top-ranked candidate selected. Interpret close scores with care.")
            with st.expander("Structured result"): st.json(result)
        except Exception as e: st.error(str(e))
with batch:
    if model_type=="wsd_outputs":
        st.write("CSV columns: `sentence`, `target_word`, and `candidate_definitions` (semicolons between candidates; each candidate is `sense_id | optional definition`). An optional `occurrence` column is one-based.")
    else:
        st.write("CSV columns: `sentence`, `target_word`, and `candidate_definitions` (definitions separated by `|`). An optional `occurrence` column is one-based.")
    up=st.file_uploader("Upload CSV",type=["csv"])
    if up and st.button("Process batch"):
        try:
            data=pd.read_csv(up); required={"sentence","target_word","candidate_definitions"}
            if not required.issubset(data.columns): raise ValueError(f"Missing columns: {sorted(required-set(data.columns))}")
            model=predictor(); out=[]
            for idx,row in data.iterrows():
                try:
                    occ=int(row.get("occurrence",1))-1 if pd.notna(row.get("occurrence",1)) else 0
                    candidate_separator=";" if model_type=="wsd_outputs" else "|"
                    candidates=parse_candidates(str(row.candidate_definitions),model_type,candidate_separator)
                    result=model.score(str(row.sentence),str(row.target_word),candidates,occ)
                    out.append({**row.to_dict(),"selected_sense":result["selected_sense"]["definition"],"selected_sense_id":result["selected_sense"]["sense_id"],"compatibility_score":result["confidence"]["score"],"warning":result["warning"],"error":""})
                except Exception as e: out.append({**row.to_dict(),"selected_sense":"","selected_sense_id":"","compatibility_score":"","warning":"","error":f"row {idx+1}: {e}"})
            result_df=pd.DataFrame(out); st.dataframe(result_df,use_container_width=True)
            st.download_button("Download results CSV",result_df.to_csv(index=False).encode("utf-8-sig"),"marathisense_results.csv","text/csv")
        except Exception as e: st.error(str(e))
with compare:
    st.write("Compare available checkpoints on the same input. To include a wsd_outputs checkpoint, enter each candidate as `sense_id | optional definition`. Each model's raw scores are shown separately and are not calibrated across models.")
    baseline_path=st.text_input("Baseline checkpoint",DEFAULT_BASELINE,key="compare_baseline")
    transformer_path=st.text_input("Transformer checkpoint","checkpoints/muril",key="compare_transformer")
    wsd_output_path=st.text_input("wsd_outputs checkpoint","wsd_outputs/models/IndicBERT-v2.pt",key="compare_wsd_outputs")
    cmp_sentence=st.text_input("Sentence",key="cmp_sentence")
    cmp_target=st.text_input("Target word",key="cmp_target")
    cmp_defs=st.text_area("Candidate senses (sense_id | optional definition, one per line)",key="cmp_defs")
    if st.button("Compare available models"):
        candidates=parse_candidates(cmp_defs,"wsd_outputs")
        choices=[("TF-IDF baseline",baseline_path,"baseline"),("Transformer",transformer_path,"transformer"),("wsd_outputs model",wsd_output_path,"wsd_outputs")]
        cols=st.columns(len(choices))
        for col,(label,path,kind) in zip(cols,choices):
            with col:
                st.subheader(label)
                resolved=checkpoint_path(path)
                if not resolved.exists(): st.warning(f"Unavailable: {resolved}"); continue
                try:
                    result=load_predictor(str(resolved),kind,int(max_length),float(threshold)).score(cmp_sentence,cmp_target,candidates)
                    st.write("Selected definition:",result["selected_sense"]["definition"])
                    st.dataframe(pd.DataFrame(result["ranked_candidates"]),use_container_width=True,hide_index=True)
                except Exception as e: st.error(str(e))
with wic:
    st.write("Compare two Marathi contexts and estimate whether the target lemma is used in the same sense. This is the SenWiCh Word-in-Context benchmark, separate from sentence–gloss ranking.")
    wic_path=st.text_input("SenWiCh checkpoint path","checkpoints/senwich_marathi_wic")
    wic_lemma=st.text_input("Target lemma",key="wic_lemma")
    wic_pos=st.text_input("Part of speech (optional)",key="wic_pos")
    wic_cols=st.columns(2)
    with wic_cols[0]:
        wic_sentence1=st.text_area("Context 1",key="wic_sentence1",height=150)
        wic_surface1=st.text_input("Target as written in context 1 (optional)",key="wic_surface1",help="If provided and found exactly, the first occurrence is marked for the model.")
    with wic_cols[1]:
        wic_sentence2=st.text_area("Context 2",key="wic_sentence2",height=150)
        wic_surface2=st.text_input("Target as written in context 2 (optional)",key="wic_surface2",help="If provided and found exactly, the first occurrence is marked for the model.")
    if st.button("Compare senses",type="primary",key="compare_wic"):
        try:
            resolved=checkpoint_path(wic_path)
            if not resolved.exists(): raise FileNotFoundError(f"Checkpoint not found: {resolved}.")
            result=load_wic_predictor(str(resolved),256).predict(wic_lemma,wic_sentence1,wic_sentence2,wic_pos,wic_surface1,wic_surface2)
            st.subheader("Prediction: "+result["prediction"])
            st.metric("Probability of same sense",f"{result['same_sense_probability']:.1%}")
            st.caption(result["warning"])
        except Exception as e: st.error(str(e))
with info:
    st.subheader("Available artifacts")
    st.write({"selected_model_type":model_type,"checkpoint":str(checkpoint_path(model_path)),"available":checkpoint_path(model_path).exists()})
    config_path=ROOT/"configs/config.yaml"
    if config_path.exists():
        import yaml
        st.subheader("Configured experiment")
        st.json(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    metrics=ROOT/"outputs/metrics/model_comparison.json"
    if metrics.exists(): st.json(json.loads(metrics.read_text(encoding="utf-8")))
    else: st.warning("No measured model report exists yet. Training is pending until the dataset files are available.")
    imported_results=ROOT/"wsd_outputs/results_table.csv"
    if imported_results.exists():
        st.subheader("Imported wsd_outputs results")
        st.caption("Reported scores from the imported experiment; its split and task setup differ from the local model comparison above.")
        st.dataframe(pd.read_csv(imported_results),use_container_width=True,hide_index=True)
    wic_metrics=ROOT/"outputs/senwich_kaggle/senwich_marathi_wic/metrics.json"
    if wic_metrics.exists():
        st.subheader("SenWiCh Marathi WiC evaluation")
        wic_report=json.loads(wic_metrics.read_text(encoding="utf-8"))
        st.json({"test_metrics":wic_report.get("test_metrics"),"split_rows":wic_report.get("split_rows"),"model":wic_report.get("model")})
    st.caption("Binary pair classification and grouped top-1 ranking are distinct evaluations.")

