from detector.score_filter import ScoreFilter # type: ignore

scores = [
    0.01,
    0.02,
    0.03,
    0.60,
    0.02,
    0.01,
]

filter = ScoreFilter(alpha=0.2)

for score in scores:

    filtered = filter.update(score)

    print(
        f"Raw={score:.3f}   Filtered={filtered:.3f}"
    )