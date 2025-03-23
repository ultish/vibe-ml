from river import tree
from collections import defaultdict

# Initialize the HoeffdingTreeClassifier with a low grace period for fast adaptation
bits_model = tree.HoeffdingTreeClassifier(grace_period=1)

# Data structures
historical_data = defaultdict(lambda: defaultdict(list))  # {source: {metric: [values]}}
thresholds = defaultdict(lambda: defaultdict(float))     # {source: {metric: min_threshold}}

def process_bits_per_sec(source, value, label=None, window_size=1000):
    # Default minimum threshold if not set
    if source not in thresholds or "bits_per_sec" not in thresholds[source]:
        thresholds[source]["bits_per_sec"] = 500000  # 500 kbps default
    
    # Get source-specific minimum threshold
    min_threshold = thresholds[source]["bits_per_sec"]
    
    # Feature: ratio of current value to minimum threshold
    relative_to_min = value / min_threshold
    
    features = {
        "relative_to_min": relative_to_min
    }

    # Predict if enough data (after 5 samples)
    prediction = None
    if len(historical_data[source]["bits_per_sec"]) > 5:
        prediction = bits_model.predict_one(features)
        print(f"{source}: {value/1000:.0f} kbps (rel_min: {relative_to_min:.2f}) -> {prediction}")
    else:
        print(f"{source}: {value/1000:.0f} kbps (Training phase)")

    # Learn if a label is provided
    if label:
        bits_model.learn_one(features, label)
        print(f"  Labeled as {label}")

    # Update historical data (kept for consistency, though not used in features)
    historical_data[source]["bits_per_sec"].append(value)
    if len(historical_data[source]["bits_per_sec"]) > window_size:
        historical_data[source]["bits_per_sec"].pop(0)

    return prediction

# Pre-train with synthetic data
print("=== Pre-Training Model with Synthetic Data ===")
# "bad" values (relative_to_min < 0.5)
for _ in range(10):
    process_bits_per_sec("dummy", 10, "bad")      # rel_min = 0.00002
for _ in range(10):
    process_bits_per_sec("dummy", 100, "bad")     # rel_min = 0.0002
for _ in range(10):
    process_bits_per_sec("dummy", 250000, "bad")  # rel_min = 0.5
# "average" values (0.5 <= relative_to_min < 1.5)
for _ in range(10):
    process_bits_per_sec("dummy", 500000, "average")  # rel_min = 1.0
for _ in range(10):
    process_bits_per_sec("dummy", 750000, "average")  # rel_min = 1.5
# "good" values (relative_to_min >= 1.5)
for _ in range(10):
    process_bits_per_sec("dummy", 1000000, "good")    # rel_min = 2.0
for _ in range(10):
    process_bits_per_sec("dummy", 2000000, "good")    # rel_min = 4.0

# Test Case 1: Source A - Historical 10 bps, Spike to 1 Mbps
print("\n=== Test Case 1: Source A - Historical 10 bps, Spike to 1 Mbps ===")
for _ in range(10):
    process_bits_per_sec("source_a", 10)          # rel_min = 0.00002
process_bits_per_sec("source_a", 1000000, "good")  # rel_min = 2.0
process_bits_per_sec("source_a", 1000000)         # Expect "good"
process_bits_per_sec("source_a", 10)              # Expect "bad"