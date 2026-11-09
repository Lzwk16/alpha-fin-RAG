"""Alpha Audit web page: type a question, get a cited answer. Start it with `uv run main.py ui`.

Runs the same pipeline as `main.py ask` (Generator: retrieve -> gate -> generate -> guards). The search models load
once per server, not once per question.
"""

import streamlit as st

from main import answer_dict

STATUS = {
    "ANSWERED": (st.success, "Answered"),
    "PARTIAL": (st.info, "Partly answered: the articles cover only part of this question"),
    "NOT_IN_CORPUS": (st.warning, "Not covered by the articles"),
    "DECLINED_ADVICE": (st.info, "Personal advice declined"),
}


def md(text: str) -> str:
    """Model and article text, shown literally: Streamlit would otherwise read "$...$" as maths."""
    return text.replace("$", "\\$")


@st.cache_resource(show_spinner="Loading the search models...")
def generator():
    from alpha_audit.llm.generator import Generator

    return Generator()


st.set_page_config(page_title="Alpha Audit", page_icon="🔎")
st.title("Alpha Audit")
st.caption("Ask about the Picture Perfect Portfolios articles. Answers cite the articles they come from. "
           "Informational only, not investment advice.")

with st.form("ask"):
    question = st.text_input("Question", placeholder="What is the net expense ratio of the QVAL ETF?")
    submitted = st.form_submit_button("Ask", type="primary")

if submitted and not question.strip():
    st.warning("Type a question first.")
elif submitted:
    gen = generator()
    with st.spinner("Searching the articles and writing an answer..."):
        try:
            d = answer_dict(gen.ask(question.strip()))
        except Exception as e:  # noqa: BLE001 -- show any failure (missing .env credentials, endpoint down) on the page
            st.error(f"Couldn't answer this question: {e}")
            st.stop()

    show, label = STATUS.get(d["status"], (st.info, d["status"]))
    show(label)
    st.markdown(md(d["answer"]))
    if d["citations"]:
        st.subheader("Sources")
        for c in d["citations"]:
            quote = md(c["quote"]).replace("\n", "\n> ")
            st.markdown(f"**{c['evidence_id']}** · {md(c['source'])}\n\n> {quote}")
    for c in d["conflicts"]:
        positions = "; ".join(f'{p["evidence_id"]}: "{p["quote"]}"' for p in c["positions"])
        st.warning(md(f"Articles disagree on {c['topic']}: {positions} (likely reason: {c['likely_reason']})"))
    for f in d["flags"]:
        st.warning(md(f"Check flagged ({f['guard']}/{f['kind']}): {f['detail']}"))
    st.caption(md(d["attribution"]))
    with st.expander("Details"):
        st.json({k: d[k] for k in ("prompt_version", "model", "gated")})
