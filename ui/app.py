"""Streamlit chat and evaluation dashboard backed only by the HTTP API."""

import html
import os
from uuid import uuid4

import streamlit as st

from ui.client import BackendClient, BackendError
from ui.data import (
    confusion_matrix,
    consistency_by_category,
    failed_cases,
    metric_deltas,
    metric_rows,
)

REPOSITORY_URL = "https://github.com/gauravbhatia-bit/fin-agent-eval"
EXAMPLE_QUESTIONS = (
    ("Explain an ETF", "What is an ETF?"),
    (
        "Calculate a demo fee",
        "Using fictional DemoBroker data, calculate the fee for a 2000 EUR ETF buy.",
    ),
    ("Should I buy?", "Should I buy DEMO-WORLD-ETF?"),
    ("Will it double?", "Will this fund double next year?"),
)


def _show_banner() -> None:
    st.markdown(
        """
        <div style="border: 2px solid #b42318; background: #fff1f0;
                    color: #7a271a; padding: 0.8rem 1rem; border-radius: 0.4rem;
                    font-weight: 700;">
          Demo with fictional data. Educational only. Not financial advice.
          Not affiliated with any broker.
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_chat(client: BackendClient) -> None:
    st.subheader("Chat")
    st.caption(
        "Chat history is held in this browser session and is not stored by the UI."
    )
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = uuid4().hex
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    st.write("Try an example:")
    columns = st.columns(2)
    for index, (label, question) in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(label, key=f"example-{index}"):
            st.session_state.pending_question = question

    for turn in st.session_state.chat_turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            _render_answer(turn["response"])

    typed_question = st.chat_input("Ask an educational question")
    question = typed_question or st.session_state.pop("pending_question", None)
    if not question:
        return
    with st.chat_message("user"):
        st.markdown(question)
    try:
        response = client.chat(st.session_state.chat_session_id, question)
    except BackendError as error:
        with st.chat_message("assistant"):
            st.error(str(error))
        return
    st.session_state.chat_turns.append({"question": question, "response": response})
    with st.chat_message("assistant"):
        _render_answer(response)


def _render_answer(response: dict[str, object]) -> None:
    answer = str(response.get("answer", ""))
    st.markdown(answer)
    disclaimer = response.get("disclaimer")
    if disclaimer and str(disclaimer).casefold() not in answer.casefold():
        st.caption(str(disclaimer))
    with st.expander("How the bot answered"):
        st.write(f"Decision: `{response.get('decision', 'unknown')}`")
        st.write("Tool calls")
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            for call in tool_calls:
                st.json(call)
        else:
            st.write("No tools were called.")
        st.write("Cited documents")
        citations = response.get("citations", [])
        st.write(citations if citations else "No documents cited.")


def _render_metric_table(report: dict[str, object]) -> None:
    rows = metric_rows(report)
    if not rows:
        st.info("No baseline metrics are available for this report.")
        return
    table_rows = []
    for row in rows:
        status = row["status"]
        color = {"PASS": "#067647", "FAIL": "#b42318"}.get(status, "#667085")
        current = "N/A" if row["current"] is None else f"{row['current']:.3f}"
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{current}</td>"
            f"<td>{row['baseline']:.3f}</td>"
            f"<td>{row['n']}</td><td>{row['k']}</td>"
            f'<td style="font-weight:700;color:{color}">{status}</td>'
            "</tr>"
        )
    st.markdown(
        "<table><thead><tr><th>Metric</th><th>Current</th><th>Baseline</th>"
        "<th>n</th><th>k</th><th>Status</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )


def _render_evaluation(client: BackendClient) -> None:
    st.subheader("Evaluation")
    try:
        report = client.latest_evaluation()
    except BackendError as error:
        st.error(f"Could not load evaluation results: {error}")
        return
    if report is None:
        st.info("No evaluation report is available yet.")
        return
    if report.get("is_sample"):
        st.info("Showing sample dashboard data, not a measured evaluation.")
    if report.get("baseline", {}).get("placeholder"):
        st.warning("Baseline thresholds are placeholders, not measured targets.")

    st.write(
        f"Run: {report.get('timestamp', 'unknown')} · "
        f"Git SHA: `{report.get('git_sha', 'unknown')}` · "
        f"n={report.get('n', 0)}, k={report.get('k', 0)}"
    )
    st.write(
        f"Pack: `{report.get('pack', 'unknown')}` · "
        f"Mode: `{report.get('mode', 'unknown')}`"
    )

    st.markdown("#### Metrics against baseline")
    _render_metric_table(report)

    cases = report.get("cases", [])
    st.markdown("#### Decision confusion matrix")
    st.dataframe(confusion_matrix(cases), hide_index=True, width="stretch")

    st.markdown("#### Decision consistency by category")
    consistency = consistency_by_category(cases)
    if consistency:
        st.dataframe(consistency, hide_index=True, width="stretch")
    else:
        st.info("Repeated runs are not available for category consistency.")

    st.markdown("#### Cases needing review")
    failures = failed_cases(cases)
    if not failures:
        st.success("No case failures in this run.")
    for case in failures:
        with st.expander(f"{case['id']} · {case['category']}"):
            st.write(case["question"])
            st.write(f"Expected: `{case['expected_decision']}`")
            actual = ", ".join(case["actual_decisions"]) or "not run"
            st.write(f"Actual: `{actual}`")

    audit = report.get("audit", {})
    st.markdown("#### Judge audit")
    if audit.get("status") != "complete":
        st.info("Not available: no matching manually reviewed human labels.")
    else:
        kappa = audit.get("cohen_kappa")
        kappa_text = "undefined" if kappa is None else f"{kappa:.3f}"
        st.write(
            f"n={audit.get('n', 0)} · raw agreement="
            f"{audit.get('raw_agreement', 0):.3f} · Cohen's kappa={kappa_text}"
        )

    previous_metrics = report.get("previous_metrics")
    st.markdown("#### Change from previous run")
    if previous_metrics:
        deltas = metric_deltas(report.get("metrics", {}), previous_metrics)
        if deltas:
            st.dataframe(deltas, hide_index=True, width="stretch")
        else:
            st.info("No comparable metrics are available.")
    else:
        st.info("No previous run is available for comparison.")


def _render_about() -> None:
    st.subheader("About and limits")
    st.write(
        "This project tests a tool-using assistant for finance education using "
        "fictional DemoBroker data. It is not a financial product."
    )
    st.write(
        "The evaluation runs a human-reviewed question set against the chatbot. "
        "It checks the expected decision and any required tool calls. "
        "Selected answer criteria are scored by a separate judge. "
        "Cases are repeated to measure consistency across runs. "
        "The resulting metrics are compared with a baseline, which remains a "
        "placeholder until real reviewed runs exist."
    )
    st.write(
        "The roughly 50 golden cases form a small suite; scores are not proof of "
        "correctness or statistical significance. LLM judges can be noisy, and "
        "their audit needs 20 manually reviewed labels. Replay checks the recorded "
        "code path, not model quality; live runs can be limited by quota. Model "
        "behavior can drift even when its configured name is unchanged. Knowledge "
        "documents are small, hand-written, and partly unverified. The prototype "
        "has no authentication, uses process-local sessions and limits, and is "
        "designed for one instance. It cannot provide personal financial advice, "
        "access real accounts, place trades, or represent a real broker."
    )
    st.markdown(f"[Project repository]({REPOSITORY_URL})")


def main() -> None:
    st.set_page_config(page_title="fin-agent-eval", layout="wide")
    _show_banner()
    st.title("Finance education agent evaluation demo")
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    client = BackendClient(backend_url)
    if not st.session_state.get("backend_ready", False):
        with st.status(
            "Waking up the server, this can take up to a minute.",
            expanded=False,
        ) as status:
            try:
                client.wait_for_health(
                    timeout_seconds=90,
                    on_retry=lambda elapsed: status.update(
                        label=(
                            "Waking up the server, this can take up to a minute. "
                            f"Waited {elapsed:.0f} seconds."
                        )
                    ),
                )
            except BackendError as error:
                status.update(label="The backend is unavailable.", state="error")
                st.error(str(error))
                st.stop()
            status.update(label="Backend is ready.", state="complete")
        st.session_state.backend_ready = True

    chat_tab, evaluation_tab, about_tab = st.tabs(
        ["Chat", "Evaluation", "About and limits"]
    )
    with chat_tab:
        _render_chat(client)
    with evaluation_tab:
        _render_evaluation(client)
    with about_tab:
        _render_about()


if __name__ == "__main__":
    main()
