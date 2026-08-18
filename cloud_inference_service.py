"""
Cloud inference service for the printer fault-detection model.

Subscribes to a HiveMQ (or any MQTT broker) topic where the ESP32 publishes
computed feature vectors, runs them through the trained model, and publishes
the prediction back to a status topic.

Run this as a small always-on process (a $5/mo VPS, a Docker container, or a
low-cost serverless-with-persistent-connection setup all work).

Environment variables expected (set these, don't hardcode credentials):
    MQTT_HOST       e.g. "xxxxxxxx.s1.eu.hivemq.cloud"
    MQTT_PORT       8883 (TLS) is HiveMQ Cloud's default
    MQTT_USERNAME
    MQTT_PASSWORD
    FEATURES_TOPIC  e.g. "printer/+/features"   (+ wildcards the printer id)
    STATUS_TOPIC_PREFIX  e.g. "printer/"        (status published to printer/{id}/status)

Usage:
    pip install paho-mqtt joblib scikit-learn pandas numpy
    MQTT_HOST=... MQTT_USERNAME=... MQTT_PASSWORD=... python cloud_inference_service.py
"""

import os
import json
import joblib
import numpy as np
import paho.mqtt.client as mqtt

MODEL_PATH = "printer_fault_model.joblib"

MQTT_HOST = os.environ.get("MQTT_HOST", "xxxxxxxx.s1.eu.hivemq.cloud")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "8883"))
MQTT_USERNAME = os.environ.get("MQTT_USERNAME", "")
MQTT_PASSWORD = os.environ.get("MQTT_PASSWORD", "")
FEATURES_TOPIC = os.environ.get("FEATURES_TOPIC", "printer/+/features")
STATUS_TOPIC_PREFIX = os.environ.get("STATUS_TOPIC_PREFIX", "printer/")

# how many consecutive fault predictions in a row before we call it a real
# alert, rather than a single noisy misfire -- tune this once you see live
# behaviour. Keyed per printer id so multiple devices don't share state.
CONSECUTIVE_THRESHOLD = 3
_consecutive_fault_counts: dict[str, int] = {}


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
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"Bad JSON on {msg.topic}, skipping")
        return

    # topic shape: printer/{printer_id}/features
    parts = msg.topic.split("/")
    printer_id = parts[1] if len(parts) > 1 else "unknown"

    try:
        result = predict(payload)
    except ValueError as e:
        print(f"[{printer_id}] {e}")
        return

    # simple debounce: only escalate to an "alert" after N consecutive
    # fault predictions in a row, to avoid reacting to a single noisy window
    if result["is_fault"]:
        _consecutive_fault_counts[printer_id] = _consecutive_fault_counts.get(printer_id, 0) + 1
    else:
        _consecutive_fault_counts[printer_id] = 0
    result["consecutive_fault_windows"] = _consecutive_fault_counts[printer_id]
    result["alert"] = _consecutive_fault_counts[printer_id] >= CONSECUTIVE_THRESHOLD

    status_topic = f"{STATUS_TOPIC_PREFIX}{printer_id}/status"
    client.publish(status_topic, json.dumps(result), qos=1)

    tag = "ALERT" if result["alert"] else ("fault?" if result["is_fault"] else "ok")
    print(f"[{printer_id}] {tag:6s} -> {result['predicted_label']:16s} "
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
