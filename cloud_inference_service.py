"""
Cloud inference service for the printer fault-detection model (PrintPulse).

Subscribes to the HiveMQ topics where the ESP32 publishes vibration features
and the PC publishes temperature readings, merges them, runs the trained
model, and publishes a backend-compatible prediction to a status topic.

Includes a minimal HTTP health-check endpoint (Flask, port 7860) so this can
run on Hugging Face Spaces: free Spaces sleep after a period of no incoming
HTTP traffic, so an external uptime pinger (e.g. UptimeRobot, cron-job.org)
hitting this endpoint every few minutes keeps the Space -- and therefore the
MQTT listener -- awake continuously.

Credentials are read from environment variables -- never hardcode them here,
even in a private repo. Repo visibility can change, collaborators can be
added, and forks/clones outlive access decisions.
"""

import json
import os
import sys
import threading
import time
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, jsonify

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
        f"Set them in your host's environment/secrets settings before running."
    )

# --- Topics ---
FEATURES_TOPIC = "printpulse/features"   # ESP32: vibration-derived features
PRINTER_TOPIC = "printpulse/printer"     # PC: nozzle_actual/bed_actual temp readings
STATUS_TOPIC = "printpulse/status"       # published prediction, consumed by backend
PRINTER_ID = "printpulse_esp32"

CONSECUTIVE_THRESHOLD = 3
_consecutive_fault_count = 0

_latest_temp = {"nozzle_temp": None, "bed_temp": None, "nozzle_target": None, "bed_target": None}
_TEMP_STALE_AFTER_S = 30
_latest_temp_time = 0

OVERHEAT_MARGIN_C = 15.0
OVERHEAT_SUSTAIN_S = 10.0
UNDERHEAT_MARGIN_C = 15.0
UNDERHEAT_SUSTAIN_S = 90.0
FALLBACK_NOZZLE_MAX_C = 260.0
FALLBACK_BED_MAX_C = 100.0

_thermal_deviation_start = {
    "nozzle_overheat": None, "nozzle_underheat": None,
    "bed_overheat": None, "bed_underheat": None,
}
_service_state = {
    "mqtt_connected": False,
    "last_message_time": None,
    "last_prediction": None,
    "messages_received": 0,
    "started_at": time.time(),
}
_state_lock = threading.Lock()

# --- Maps model output labels to the backend's FaultClass enum values ---
# Backend (app/models.py) only accepts exactly: NORMAL, NOZZLE_CLOG,
# MOTOR_FAULT, THERMAL_RUNAWAY. Update the keys on the left if your model's
# bundle['labels'] print different exact strings than these.
_FAULT_CLASS_MAP = {
    "normal": "NORMAL",
    "nozzle_clog": "NOZZLE_CLOG",
    "motor_fault": "MOTOR_FAULT",
    "thermal_runaway": "THERMAL_RUNAWAY",
}


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


def to_backend_payload(result: dict) -> dict:
    """Maps the internal prediction result to the exact shape the backend's
    MQTTPayload schema expects (app/schemas.py). accel_rms_z/temperature are
    approximated below -- adjust if you'd rather send different fields."""
    label_key = str(result["predicted_label"]).strip().lower().replace(" ", "_")
    fault_class = _FAULT_CLASS_MAP.get(label_key)
    if fault_class is None:
        print(f"WARNING: unmapped label '{result['predicted_label']}', defaulting to NORMAL")
        fault_class = "NORMAL"

    return {
        "fault_class": fault_class,
        "confidence": result["confidence"],
        "accel_rms_z": result.get("head_rms"),        # approximation: combined-magnitude RMS, not pure Z-axis
        "current_rms": None,                           # no current sensor in this build
        "temperature": result.get("nozzle_temp"),       # nozzle chosen over bed as the more diagnostic reading
        "timestamp": int(time.time() * 1000),
    }


def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to {MQTT_HOST}. Subscribing to {FEATURES_TOPIC} and {PRINTER_TOPIC}")
        client.subscribe(FEATURES_TOPIC, qos=1)
        client.subscribe(PRINTER_TOPIC, qos=1)
        with _state_lock:
            _service_state["mqtt_connected"] = True
    else:
        print(f"Connection failed, rc={rc}")
        with _state_lock:
            _service_state["mqtt_connected"] = False


def on_disconnect(client, userdata, rc, properties=None):
    with _state_lock:
        _service_state["mqtt_connected"] = False
    print(f"Disconnected from MQTT (rc={rc}), paho will auto-reconnect")


def handle_printer_message(payload: dict):
    global _latest_temp_time
    if "nozzle_actual" in payload and "bed_actual" in payload:
        _latest_temp["nozzle_temp"] = payload["nozzle_actual"]
        _latest_temp["bed_temp"] = payload["bed_actual"]
        _latest_temp["nozzle_target"] = payload.get("nozzle_target")
        _latest_temp["bed_target"] = payload.get("bed_target")
        _latest_temp_time = time.time()
    else:
        print(f"[{PRINTER_ID}] printer topic message missing nozzle_actual/bed_actual: {payload}")


def _check_one_sensor(actual, target, overheat_key, underheat_key, fallback_max):
    now = time.time()

    if target is not None and target > 0:
        if actual > target + OVERHEAT_MARGIN_C:
            _thermal_deviation_start[underheat_key] = None
            start = _thermal_deviation_start[overheat_key]
            if start is None:
                _thermal_deviation_start[overheat_key] = now
                return None
            if now - start >= OVERHEAT_SUSTAIN_S:
                return f"overheating (actual {actual:.1f} vs target {target:.1f})"
            return None

        elif actual < target - UNDERHEAT_MARGIN_C:
            _thermal_deviation_start[overheat_key] = None
            start = _thermal_deviation_start[underheat_key]
            if start is None:
                _thermal_deviation_start[underheat_key] = now
                return None
            if now - start >= UNDERHEAT_SUSTAIN_S:
                return f"underheating/heater failure (actual {actual:.1f} vs target {target:.1f})"
            return None

        else:
            _thermal_deviation_start[overheat_key] = None
            _thermal_deviation_start[underheat_key] = None
            return None
    else:
        if actual > fallback_max:
            start = _thermal_deviation_start[overheat_key]
            if start is None:
                _thermal_deviation_start[overheat_key] = now
                return None
            if now - start >= OVERHEAT_SUSTAIN_S:
                return f"overheating (actual {actual:.1f} exceeds fallback ceiling {fallback_max:.1f}, no target available)"
            return None
        _thermal_deviation_start[overheat_key] = None
        return None


def check_thermal_anomaly() -> dict:
    nozzle_reason = _check_one_sensor(
        _latest_temp["nozzle_temp"], _latest_temp["nozzle_target"],
        "nozzle_overheat", "nozzle_underheat", FALLBACK_NOZZLE_MAX_C,
    )
    bed_reason = _check_one_sensor(
        _latest_temp["bed_temp"], _latest_temp["bed_target"],
        "bed_overheat", "bed_underheat", FALLBACK_BED_MAX_C,
    )
    reasons = [r for r in (nozzle_reason, bed_reason) if r]
    return {"thermal_anomaly": len(reasons) > 0, "thermal_reasons": reasons}


def handle_features_message(client, payload: dict):
    global _consecutive_fault_count

    temp_age = time.time() - _latest_temp_time if _latest_temp_time else None
    if _latest_temp["nozzle_temp"] is None or (temp_age is not None and temp_age > _TEMP_STALE_AFTER_S):
        print(f"[{PRINTER_ID}] skipping prediction: no recent temp reading "
              f"(age={temp_age if temp_age is not None else 'never received'})")
        return

    merged = dict(payload)
    merged["nozzle_temp"] = _latest_temp["nozzle_temp"]
    merged["bed_temp"] = _latest_temp["bed_temp"]

    try:
        result = predict(merged)
    except ValueError as e:
        print(f"[{PRINTER_ID}] {e}")
        return

    result["nozzle_temp"] = merged["nozzle_temp"]
    result["bed_temp"] = merged["bed_temp"]
    result["head_rms"] = merged.get("head_rms")

    thermal = check_thermal_anomaly()
    result.update(thermal)

    if result["is_fault"]:
        _consecutive_fault_count += 1
    else:
        _consecutive_fault_count = 0
    result["consecutive_fault_windows"] = _consecutive_fault_count
    result["alert"] = (_consecutive_fault_count >= CONSECUTIVE_THRESHOLD) or thermal["thermal_anomaly"]
    result["printer_id"] = PRINTER_ID

    with _state_lock:
        _service_state["last_prediction"] = result

    # NOTE: thermal_anomaly/thermal_reasons are NOT part of the backend's
    # MQTTPayload schema, so they aren't sent below -- they're logged locally
    # only. If you want thermal-rule detections to reach the dashboard, that
    # needs its own decision (e.g. override fault_class to THERMAL_RUNAWAY
    # when thermal_anomaly is true) -- not done automatically here.
    client.publish(STATUS_TOPIC, json.dumps(to_backend_payload(result)), qos=1)

    if thermal["thermal_anomaly"]:
        tag = "ALERT"
        label_str = f"thermal_anomaly ({'; '.join(thermal['thermal_reasons'])})"
    else:
        tag = "ALERT" if result["alert"] else ("fault?" if result["is_fault"] else "ok")
        label_str = result["predicted_label"]
    print(f"[{PRINTER_ID}] {tag:6s} -> {label_str} "
          f"(conf={result['confidence']:.2f}, streak={result['consecutive_fault_windows']}, "
          f"temp_age={temp_age:.1f}s)")


def on_message(client, userdata, msg):
    with _state_lock:
        _service_state["messages_received"] += 1
        _service_state["last_message_time"] = time.time()

    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except json.JSONDecodeError:
        print(f"Bad JSON on {msg.topic}, skipping")
        return

    if msg.topic == PRINTER_TOPIC:
        handle_printer_message(payload)
    elif msg.topic == FEATURES_TOPIC:
        handle_features_message(client, payload)


app = Flask(__name__)


@app.route("/")
def health():
    with _state_lock:
        state_copy = dict(_service_state)
    state_copy["uptime_seconds"] = round(time.time() - state_copy["started_at"], 1)
    return jsonify(state_copy)


def run_flask():
    app.run(host="0.0.0.0", port=7860)


def run_mqtt():
    client = mqtt.Client(client_id="printer-fault-inference", protocol=mqtt.MQTTv5)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()


def main():
    mqtt_thread = threading.Thread(target=run_mqtt, daemon=True)
    mqtt_thread.start()
    run_flask()


if __name__ == "__main__":
    main()