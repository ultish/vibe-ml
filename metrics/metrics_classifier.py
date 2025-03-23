from river import tree
from collections import defaultdict
import numpy as np

# Initialize the HoeffdingTreeClassifier with a low grace period for fast adaptation
bits_model = tree.HoeffdingTreeClassifier(grace_period=1)

# Data structures
historical_data = defaultdict(lambda: defaultdict(list))  # {source: {metric: [values]}}
thresholds = defaultdict(lambda: defaultdict(float))     # {source: {metric: min_threshold}}

def process_bits_per_sec(source, value, label=None, window_size=1000):
    """
    Process a bps value for a given source, predict its class, and learn from a label if provided.
    
    Args:
        source (str): The source identifier (e.g., "source_a").
        value (float): The bps value to process.
        label (str, optional): The true label ("good", "average", "bad") for learning.
        window_size (int): Maximum number of historical values to retain.
    
    Returns:
        str: Predicted class ("good", "average", "bad") or None if insufficient data.
    """
    # Default minimum threshold if not set
    if source not in thresholds or "bits_per_sec" not in thresholds[source]:
        thresholds[source]["bits_per_sec"] = 500000  # 500 kbps default
    
    # Get source-specific minimum threshold
    min_threshold = thresholds[source]["bits_per_sec"]
    
    # Feature: ratio of current value to minimum threshold
    relative_to_min = value / min_threshold if min_threshold > 0 else 0
    
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

# Test Case 2: Source B - Stable at 600 kbps
print("\n=== Test Case 2: Source B - Stable at 600 kbps ===")
thresholds["source_b"]["bits_per_sec"] = 500000
for _ in range(10):
    process_bits_per_sec("source_b", 600000, "good")  # rel_min = 1.2
process_bits_per_sec("source_b", 600000)              # Expect "good"

# Test Case 3: Source C - Sudden drop from 1 Mbps to 100 bps
print("\n=== Test Case 3: Source C - Sudden drop from 1 Mbps to 100 bps ===")
thresholds["source_c"]["bits_per_sec"] = 500000
for _ in range(5):
    process_bits_per_sec("source_c", 1000000, "good")  # rel_min = 2.0
process_bits_per_sec("source_c", 100, "bad")           # rel_min = 0.0002
process_bits_per_sec("source_c", 100)                  # Expect "bad"

# Test Case 4: Source D - Fluctuating around threshold
print("\n=== Test Case 4: Source D - Fluctuating around threshold ===")
thresholds["source_d"]["bits_per_sec"] = 500000
fluctuating_values = [400000, 600000, 450000, 550000, 490000, 510000]
for val in fluctuating_values:
    process_bits_per_sec("source_d", val, "average" if 450000 <= val <= 550000 else "bad" if val < 450000 else "good")
process_bits_per_sec("source_d", 500000)  # Expect "average"

# Test Case 5: Source E - Zero bps
print("\n=== Test Case 5: Source E - Zero bps ===")
thresholds["source_e"]["bits_per_sec"] = 500000
for _ in range(5):
    process_bits_per_sec("source_e", 0, "bad")  # rel_min = 0.0
process_bits_per_sec("source_e", 0)             # Expect "bad"

# Test Case 6: Source F - Slightly below threshold
print("\n=== Test Case 6: Source F - Slightly below threshold ===")
thresholds["source_f"]["bits_per_sec"] = 500000
process_bits_per_sec("source_f", 499999, "bad")  # rel_min = 0.999998
process_bits_per_sec("source_f", 499999)         # Expect "bad"

# Test Case 7: Source G - Exactly at threshold
print("\n=== Test Case 7: Source G - Exactly at threshold ===")
thresholds["source_g"]["bits_per_sec"] = 500000
process_bits_per_sec("source_g", 500000, "average")  # rel_min = 1.0
process_bits_per_sec("source_g", 500000)             # Expect "average"

# Test Case 8: Source H - Slightly above threshold
print("\n=== Test Case 8: Source H - Slightly above threshold ===")
thresholds["source_h"]["bits_per_sec"] = 500000
process_bits_per_sec("source_h", 500100, "good")  # rel_min = 1.0002
process_bits_per_sec("source_h", 500100)          # Expect "good"

# Test Case 9: Source I - Far above threshold
print("\n=== Test Case 9: Source I - Far above threshold ===")
thresholds["source_i"]["bits_per_sec"] = 500000
process_bits_per_sec("source_i", 10000000, "good")  # rel_min = 20.0
process_bits_per_sec("source_i", 10000000)          # Expect "good"

# Test Case 10: Source J - New source with limited data
print("\n=== Test Case 10: Source J - New source with limited data ===")
thresholds["source_j"]["bits_per_sec"] = 500000
process_bits_per_sec("source_j", 600000)  # Training phase
process_bits_per_sec("source_j", 600000)  # Training phase
process_bits_per_sec("source_j", 600000)  # Training phase
process_bits_per_sec("source_j", 600000)  # Training phase
process_bits_per_sec("source_j", 600000)  # Training phase
process_bits_per_sec("source_j", 600000)  # Expect "good" after sufficient data