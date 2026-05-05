"""Streamlit dashboard for interactive evaluation result exploration."""

import json
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st


def _get_results_dir() -> Path:
    """Return the default results directory."""
    return Path(__file__).resolve().parent.parent.parent / "data" / "results"


def _load_result_files() -> List[Dict[str, Any]]:
    """Load all JSON result files from the results directory.

    Returns:
        List of (filename, parsed data) tuples.
    """
    results_dir = _get_results_dir()
    files = sorted(results_dir.glob("*.json"), reverse=True)
    results = []
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["_filename"] = f.name
                results.append(data)
        except Exception:
            pass
    return results


def main() -> None:
    """Streamlit dashboard entry point."""
    st.set_page_config(page_title="RAG Eval Dashboard", layout="wide")
    st.title("RAG Evaluation Dashboard")

    results = _load_result_files()

    if not results:
        st.warning("No evaluation results found in data/results/. Run an evaluation first.")
        return

    # Sidebar: select a run
    st.sidebar.header("Select Run")
    filenames = [r["_filename"] for r in results]
    selected = st.sidebar.selectbox("Result file", filenames)
    run_data = next(r for r in results if r["_filename"] == selected)

    # Summary
    st.header("Run Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Run ID", run_data.get("run_id", "N/A"))
    col2.metric("Test Cases", run_data.get("num_cases", 0))
    col3.metric("Total Time", f"{run_data.get('total_latency_ms', 0):.0f} ms")

    # Aggregated scores
    agg_scores = run_data.get("aggregated_scores", {})
    if agg_scores:
        st.header("Aggregated Scores")

        import pandas as pd
        df = pd.DataFrame(
            [{"Metric": k, "Score": v} for k, v in sorted(agg_scores.items())]
        )
        st.dataframe(df, use_container_width=True)

        # Bar chart
        try:
            import plotly.express as px
            fig = px.bar(
                df, x="Metric", y="Score",
                title="Metric Scores",
                color="Score",
                color_continuous_scale="RdYlGn",
            )
            fig.update_layout(yaxis_range=[0, 5])
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(df.set_index("Metric"))

    # Per-case results
    per_case = run_data.get("per_case_results", [])
    if per_case:
        st.header("Per-Case Results")

        for i, case in enumerate(per_case, 1):
            with st.expander(f"Case {i}: {case.get('question', '')[:80]}..."):
                st.markdown(f"**Question:** {case.get('question', '')}")
                st.markdown(f"**Answer:** {case.get('answer', 'N/A')}")
                st.markdown(f"**Ground Truth:** {case.get('ground_truth', 'N/A')}")

                scores = case.get("scores", {})
                if scores:
                    st.markdown("**Scores:**")
                    for metric, score in scores.items():
                        st.write(f"- {metric}: {score:.4f}")

                reasoning = case.get("judge_reasoning", {})
                if reasoning:
                    st.markdown("**Judge Reasoning:**")
                    for criterion, reason in reasoning.items():
                        st.write(f"- **{criterion}**: {reason}")

                composite = case.get("composite_score")
                if composite is not None:
                    st.metric("Composite Score", f"{composite:.4f}")

    # Comparison mode
    st.sidebar.header("Compare Runs")
    if len(results) >= 2:
        compare_with = st.sidebar.selectbox(
            "Compare with",
            [f for f in filenames if f != selected],
        )
        if st.sidebar.button("Compare"):
            other_data = next(r for r in results if r["_filename"] == compare_with)
            _show_comparison(run_data, other_data, selected, compare_with)


def _show_comparison(
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    label_a: str,
    label_b: str,
) -> None:
    """Display comparison between two runs.

    Args:
        run_a: First run data.
        run_b: Second run data.
        label_a: Label for first run.
        label_b: Label for second run.
    """
    st.header("Run Comparison")

    scores_a = run_a.get("aggregated_scores", {})
    scores_b = run_b.get("aggregated_scores", {})
    all_metrics = sorted(set(scores_a.keys()) | set(scores_b.keys()))

    import pandas as pd
    rows = []
    for metric in all_metrics:
        va = scores_a.get(metric, 0)
        vb = scores_b.get(metric, 0)
        delta = vb - va if va is not None and vb is not None else None
        rows.append({
            "Metric": metric,
            label_a: round(va, 4) if va else "N/A",
            label_b: round(vb, 4) if vb else "N/A",
            "Delta": round(delta, 4) if delta is not None else "N/A",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
