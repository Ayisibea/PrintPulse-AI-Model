"""
Cloud inference service for the printer fault-detection model (PrintPulse).

Subscribes to the HiveMQ topic where the ESP32 publishes computed feature
vectors, runs them through the trained model, and publishes the prediction
back to a status topic.

Run this as a small always-on process (Oracle Cloud free-tier VM via
systemd, or any other always-on host).

Credentials are read from environment variables (see .env.example) — never
hardcode them here, even in a private repo. Repo visibility can change,
collaborators can be added, and forks/clones outlive access decisions.
"""

import json
import os
import sys

import joblib
import numpy as np
import paho.mqtt.client as mqtt

MODEL_PATH = "printer_fault_model.joblib"

# --- HiveMQ Cloud credentials (from environment) ---
MQTT_HOST = os.environ.get("MQTT_HOST")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "8883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD")

_required = {
    "MQTT_HOST": MQTT_HOST,
    "MQTT_USERNAME": MQTT_USERNAME,
    "MQTT_PASSWORD": MQTT_PASSWORD,
}
_missing = [name for name, val in _required.items() if not val]
if _missing:
    sys.exit(
        f"Missing required environment variable(s): {', '.join(_missing)}. "
        f"Set them (e.g. via a .env file loaded by your process manager, or "
        f"`export VAR=value`) before running this service."
    )

# --- Topics: matching the ESP32 sketch exactly (single printer, flat topics) ---
FEATURES_TOPIC = "printpulse/features"
STATUS_TOPIC = "printpulse/status"
PRINTER_ID = "printpulse_esp32"  # tag for logs/status payload; no per-device wildcarding needed

# how many consecutive fault predictions in a row before we call it a real
# alert, rather than a single noisy misfire -- tune this once you see live
# behaviour.
CONSECUTIVE_THRESHOLD = 3

_consecutive_fault_count = 0


def load_model_bundle(path: str = MODEL_PATH):
    bundle = joblib.load(path)
    print(f"Loaded model. Classes: {bundle['labels']}")
    print(f"Expecting {len(bundle['feature_cols'])} features: {bundle['feature_cols']}")
    return bundle


BUNDLE = load_model_bundle()


def predict(feature_dict: dict) -> dict:
    """feature_dict must contain (at least) all keys in BUNDLE['feature_cols']."""
    missing = [c for c in BUNDLE["feature_cols"] if c not in feature_dict]
    if missing:
        raise ValueError(f"Missing features in payload: {missing}")

    x = np.array([[feature_dict[c] for c in BUNDLE["feature_cols"]]])
    x_scaled = BUNDLE["scaler"].transform(x)
    pred_label = BUNDLE["model"].predict(x_scaled)[0]
    proba = BUNDLE["model"].predict_proba(x_scaled)[0]
    class_probs = dict(zip(BUNDLE["model"].classes_, proba.round(3)))

    return {
        "predicted_label": pred_label,
        "confidence": float(max(proba)),
        "class_probabilities": {k: float(v) for k, v in class_probs.items()},
        "is_fault": pred_label != "normal",
    }


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to {MQTT_HOST}. Subscribing to {FEATURES_TOPIC}")
        client.subscribe(FEATURES_TOPIC, qos=1)
    else:
        print(f"Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    global _consecutive_fault_count
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"Bad JSON on {msg.topic}, skipping")
        return

    try:
        result = predict(payload)
    except ValueError as e:
        print(f"[{PRINTER_ID}] {e}")
        return

    # simple debounce: only escalate to an "alert" after N consecutive
    # fault predictions in a row, to avoid reacting to a single noisy window
    if result["is_fault"]:
        _consecutive_fault_count += 1
    else:
        _consecutive_fault_count = 0

    result["consecutive_fault_windows"] = _consecutive_fault_count
    result["alert"] = _consecutive_fault_count >= CONSECUTIVE_THRESHOLD
    result["printer_id"] = PRINTER_ID

    client.publish(STATUS_TOPIC, json.dumps(result), qos=1)

    tag = "ALERT" if result["alert"] else ("fault?" if result["is_fault"] else "ok")
    print(f"[{PRINTER_ID}] {tag:6s} -> {result['predicted_label']:16s} "
          f"(conf={result['confidence']:.2f}, streak={result['consecutive_fault_windows']})")


def main():
    client = mqtt.Client(client_id="printer-fault-inference", protocol=mqtt.MQTTv5)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()  # HiveMQ Cloud requires TLS on port 8883
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
