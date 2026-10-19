"""results/summary.md and results/retrieval.png from an evaluation run."""

from pathlib import Path


def write_report(out: Path, retrieval: dict, answers: dict = None, meta: dict = None):
    lines = ["## Retrieval", "",
             "| retriever | Recall@1 | Recall@5 | Recall@10 | MRR@10 | right paper @5 | median latency |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in retrieval.items():
        lines.append(f"| {name} | {m['recall@1']:.2f} | {m['recall@5']:.2f} | {m['recall@10']:.2f} | "
                     f"{m['mrr@10']:.2f} | {m['paper@5']:.2f} | {m['latency_ms_median']:.0f} ms |")
    if answers:
        a = answers
        lines += ["", "## Answers", "",
                  "| | RAG | closed book |", "|---|---:|---:|",
                  f"| correctness (1-5, LLM judge) | {a['rag_correctness']:.2f} | {a['closed_book_correctness']:.2f} |",
                  "", f"- supported by sources: {a['faithful_supported']:.0%} "
                      f"(partial {a['faithful_partial']:.0%}, unsupported {a['faithful_unsupported']:.0%})",
                  f"- answers citing the gold passage: {a['cites_gold_rate']:.0%}",
                  f"- abstained (\"sources don't cover this\"): {a['abstention_rate']:.0%}",
                  f"- answers with an invalid citation number: {a['invalid_citation_rate']:.0%}",
                  f"- questions: {a['n']}"
                  + (f" ({a['judge_failures']} with an unreadable judge reply, left out of the averages)"
                     if a.get("judge_failures") else "")]
    if meta:
        lines += ["", "## Setup", ""] + [f"- {k}: {v}" for k, v in meta.items()]
    (out / "summary.md").write_text("\n".join(lines) + "\n")

    try:
        _plot(out / "retrieval.png", retrieval)
    except ImportError:
        pass


def _plot(path: Path, retrieval: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    colors = {"recall@1": "#2a78d6", "recall@5": "#eb6834", "recall@10": "#1baf7a"}
    names = list(retrieval)
    fig, ax = plt.subplots(figsize=(7.5, 3.6), facecolor=surface)
    ax.set_facecolor(surface)
    w = 0.26
    for j, (metric, color) in enumerate(colors.items()):
        xs = [i + (j - 1) * w for i in range(len(names))]
        vals = [retrieval[n][metric] for n in names]
        ax.bar(xs, vals, width=w - 0.03, color=color, label=metric.replace("recall", "Recall"))
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=8, color=ink2)
    ax.set_xticks(range(len(names)), names, color=ink2)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("share of questions", color=ink2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.grid(axis="y", color=grid)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=ink2, ncol=3, loc="upper left")
    ax.set_title("Is the passage the question came from retrieved?", color=ink, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
