"""Quick functional verification of the prediction pipeline."""
import sys
sys.path.insert(0, ".")

from src.predictor import predict_single, predict_batch

# Test 1: Positive review
r1 = predict_single("This product is absolutely amazing! Best purchase ever.")
print("TEST 1 - Positive review:")
print("  label={} confidence={:.4f} valid={}".format(r1["label"], r1["confidence"], r1["input_valid"]))

# Test 2: Negative review
r2 = predict_single("Terrible quality, broke after two days. Complete waste of money.")
print("TEST 2 - Negative review:")
print("  label={} confidence={:.4f} valid={}".format(r2["label"], r2["confidence"], r2["input_valid"]))

# Test 3: Empty input
r3 = predict_single("")
print("TEST 3 - Empty input:")
print("  valid={} error={}".format(r3["input_valid"], r3["error"]))

# Test 4: Too long
r4 = predict_single("x" * 15000)
print("TEST 4 - Too long input:")
print("  valid={}".format(r4["input_valid"]))

# Test 5: Batch
texts = ["Great product!", "Terrible!", "", "Okay I guess"]
batch = predict_batch(texts)
print("TEST 5 - Batch prediction:")
for _, row in batch.iterrows():
    status = "VALID" if row.is_valid else "SKIP "
    print("  [{}] {:<30s} -> {}".format(status, str(row.review_text)[:30], row.predicted_sentiment))

# Test 6: Probabilities sum to 1
r5 = predict_single("Excellent product, highly recommended.")
total = sum(r5["probabilities"].values())
print("TEST 6 - Probabilities sum to 1: {} (diff={:.8f})".format(abs(total - 1.0) < 1e-6, abs(total - 1.0)))

# Test 7: Local feature interpretability
r7 = predict_single("I love this product, it works great and very happy!")
print("TEST 7 - Feature interpretability:")
if r7.get("explanation") and r7["explanation"].get("signals"):
    terms = [s["term"] for s in r7["explanation"]["signals"]]
    print("  driving signals: {}".format(terms))
else:
    print("  explanation summary: {}".format(r7.get("explanation", {}).get("summary")))

# Test 8: Placeholder input validation
r8 = predict_single("nan")
print("TEST 8 - Placeholder input rejected: {}".format(not r8["input_valid"]))

print("\nAll functional checks complete.")
