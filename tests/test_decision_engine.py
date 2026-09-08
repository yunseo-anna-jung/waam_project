from detector.decision_engine import DecisionEngine

engine = DecisionEngine(

    warning_threshold=0.30,

    danger_threshold=0.60,

)

scores = [

    0.02,

    0.12,

    0.31,

    0.41,

    0.81,

]

for score in scores:

    action = engine.update(score)

    print()

    print("=" * 40)

    print(f"Frame      : {action.frame_id}")

    print(f"Time       : {action.timestamp:.3f}")

    print(f"Score      : {action.score:.5f}")

    print(f"Level      : {action.level}")

    print(f"Label      : {action.label}")

    print(f"Save       : {action.save}")

    print(f"Warning    : {action.warning}")