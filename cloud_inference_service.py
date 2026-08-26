"""
Cloud inference service for the printer fault-detection model (PrintPulse).

Subscribes to two HiveMQ topics:
  - printpulse/features: ESP32-computed vibration feature vectors (16 keys)
  - printpulse/printer:  PC-polled nozzle/bed temperature readings

Merges the latest temperature reading into each vibration feature vector
before running a prediction, and publishes the result to a status topic.

Run this as a small always-on process (Oracle Cloud free-tier VM via
systemd, or any other always-on host).

Credentials are read from environment variables (see .env.example) — never
hardcode them here, even in a private repo. Repo visibility can change,
collaborators can be added, and forks/clones outlive access decisions.
"""

import json
import os
import sys
import time

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

# --- Topics ---
FEATURES_TOPIC = "printpulse/features"
PRINTER_TOPIC = "printpulse/printer"
STATUS_TOPIC = "printpulse/status"
PRINTER_ID = "printpulse_esp32"  # tag for logs/status payload; no per-device wildcarding needed

# how many consecutive fault predictions in a row before we call it a real
# alert, rather than a single noisy misfire -- tune this once you see live
# behaviour.
CONSECUTIVE_THRESHOLD = 3

# how old a temperature reading is allowed to be before we refuse to use it
# for a prediction (PC poller publishes roughly every ~6s, so this gives
# margin for a couple of missed cycles without silently going stale forever)
TEMP_STALENESS_LIMIT_SEC = 15

_consecutive_fault_count = 0
_latest_temps = {"nozzle_temp": None, "bed_temp": None}
_latest_temp_timestamp = 0


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
        print(f"Connected to {MQTT_HOST}. Subscribing to {FEATURES_TOPIC} and {PRINTER_TOPIC}")
        client.subscribe(FEATURES_TOPIC, qos=1)
        client.subscribe(PRINTER_TOPIC, qos=1)
    else:
        print(f"Connection failed, rc={rc}")


def on_message(client, userdata, msg):
    global _consecutive_fault_count, _latest_temp_timestamp

    if msg.topic == PRINTER_TOPIC:
        try:
            temp_payload = json.loads(msg.payload.decode("utf-8"))
            _latest_temps["nozzle_temp"] = temp_payload["nozzle_actual"]
            _latest_temps["bed_temp"] = temp_payload["bed_actual"]
            _latest_temp_timestamp = time.time()
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Bad printer payload: {e}")
        return  # nothing to predict on a temp-only message

    if msg.topic == FEATURES_TOPIC:
        if _latest_temps["nozzle_temp"] is None:
            print("No temperature reading yet, skipping prediction")
            return

        temp_age = time.time() - _latest_temp_timestamp
        if temp_age > TEMP_STALENESS_LIMIT_SEC:
            print(f"Temp reading is {temp_age:.0f}s old, skipping prediction")
            return

        try:
            vibe_payload = json.loads(msg.payload.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"Bad JSON on {msg.topic}, skipping")
            return

        feature_dict = {**vibe_payload, **_latest_temps}

        try:
            result = predict(feature_dict)
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