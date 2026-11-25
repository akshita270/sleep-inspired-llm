"""Streamlit dashboard. Every number is read from experiments/results/ (no hand-typed results).

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.express as px
import streamlit as st

from src.reporting import results_loader as R
from src.utils.config_loader import PROJECT_ROOT

st.set_page_config(page_title="Sleep-Inspired Continual Learning", layout="wide")
COLORS = {"No replay": "#d1495b", "General replay": "#edae49", "Full replay": "#00798c"}

st.title("Sleep-inspired continual learning for a local LLM")
st.caption("Qwen3-4B (full precision) + LoRA on an Apple M2, 16 GB. Wake: collect and validate. Sleep: train, evaluate, gate. "
           "All numbers below are measured and read from files.")

with st.expander("New here? Plain-English guide (higher is better, except forgetting)", expanded=True):
    for term, text in R.plain_guide():
        st.markdown(f"**{term}:** {text}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Final exam (A/B/C)", "5-cycle study", "Gate & tuning", "Stored experiences", "Limitations"])

with tab1:
    df = R.phase8_comparison()
    st.subheader("220-question exam: original vs memory vs LoRA")
    long = df.melt(id_vars="model", value_vars=["new_facts", "old_knowledge", "reasoning"], var_name="test", value_name="accuracy")
    long["accuracy"] *= 100
    st.plotly_chart(px.bar(long, x="model", y="accuracy", color="test", barmode="group", range_y=[0, 105]), width="stretch")
    show = df.copy()
    for c in ["new_facts", "old_knowledge", "reasoning", "forgetting_rate"]:
        show[c] = (show[c] * 100).round(1)
    st.dataframe(show, hide_index=True, width="stretch")
    st.info("Episodic memory (B) looks facts up and does not change the model. LoRA (C) changes the model's behaviour but forgets some old knowledge; replay reduces that.")

with tab2:
    cont = R.continual_table()
    st.subheader("30 facts taught 6 per cycle, 5 cycles, one adapter carried forward")
    metric = st.selectbox("Metric", {"acc_taught_so_far": "Facts taught so far (accuracy)",
                                     "acc_earlier_cycles_facts": "Earlier-cycle facts kept",
                                     "acc_old": "Old knowledge", "acc_reasoning": "Reasoning",
                                     "forgetting_rate": "Forgetting rate",
                                     "training_seconds": "Training seconds"}.keys(),
                          format_func=lambda k: {"acc_taught_so_far": "Facts taught so far (accuracy)",
                                                 "acc_earlier_cycles_facts": "Earlier-cycle facts kept",
                                                 "acc_old": "Old knowledge", "acc_reasoning": "Reasoning",
                                                 "forgetting_rate": "Forgetting rate", "training_seconds": "Training seconds"}[k])
    d = cont.dropna(subset=[metric]).copy()
    if metric != "training_seconds":
        d[metric] = d[metric] * 100
    fig = px.line(d, x="cycle", y=metric, color="arm_label", markers=True, color_discrete_map=COLORS)
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, width="stretch")
    st.markdown("**Does the model still know each fact group?** (rows: exam after cycle N, columns: facts taught in cycle G)")
    cols = st.columns(3)
    for col, arm in zip(cols, R.ARM_ORDER):
        m = R.fact_group_matrix(arm) * 100
        m.columns = [f"G{c}" for c in m.columns]
        col.caption(R.ARM_LABELS[arm])
        col.plotly_chart(px.imshow(m, text_auto=".0f", zmin=0, zmax=100, color_continuous_scale="YlGnBu", aspect="auto")
                         .update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=0, b=0)), width="stretch")
    st.dataframe(cont.drop(columns=["arm_label", "acc_by_fact_group"]), hide_index=True, width="stretch")
    st.warning(f"Gate verdict: {int(cont['gate_would_pass'].sum())} of {len(cont)} cycles would have been promoted. "
               "Each arm was run once, and one fact is 16.7 points inside a 6-fact group.")

with tab3:
    st.subheader("Promotion gate (rules fixed before the runs)")
    g = R.phase8_gate_checks()
    g["value"] = g["value"].astype(str)
    g["result"] = g["passed"].map({True: "pass", False: "FAIL"})
    st.dataframe(g.drop(columns="passed"), hide_index=True, width="stretch")
    st.subheader("Settings tried on the dev set (not the final exam)")
    t = R.tuning_table()
    st.dataframe(t, hide_index=True, width="stretch")

with tab4:
    import pandas as pd
    import sqlite3
    from src.utils.config_loader import load_config
    path = PROJECT_ROOT / load_config("system_config")["database"]["path"]
    st.subheader("Experiences in the real SQLite store")
    if path.exists():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        ex = pd.read_sql("SELECT * FROM experiences ORDER BY rowid DESC LIMIT 200", conn)
        cy = pd.read_sql("SELECT * FROM sleep_cycles ORDER BY rowid DESC LIMIT 50", conn)
        conn.close()
        st.write(f"{len(ex)} experiences shown")
        st.dataframe(ex, hide_index=True, width="stretch")
        st.subheader("Sleep cycles")
        st.dataframe(cy, hide_index=True, width="stretch")
        if ex.empty:
            st.info("The real database is empty: all experiments used separate throw-away databases. Use scripts/chat.py to add real experiences.")
    else:
        st.info("No database file yet. Run scripts/chat.py to create one.")

with tab5:
    st.subheader("What these results do not show")
    for line in R.known_limitations():
        st.markdown(f"- {line}")

st.divider()
st.subheader("Conclusion")
for line in R.conclusion():
    st.markdown(f"- {line}")
