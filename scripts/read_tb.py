"""Read TensorBoard event files, print latest training status."""
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = Path("outputs/logs/tb")
ea = EventAccumulator(str(logdir))
ea.Reload()

tags = ea.Tags()["scalars"]
print("Available metrics:", len(tags))
print("-" * 50)

for tag in sorted(tags):
    events = ea.Scalars(tag)
    if not events:
        continue
    latest = events[-1]
    first = events[0]
    print(f"{tag:<35} step={latest.step:>8}  current={latest.value:.6f}  first={first.value:.6f}")
