"""
Cloud inference service for the printer fault-detection model (PrintPulse).

Subscribes to the HiveMQ topic where the ESP32 publishes computed feature
vectors, runs them through the trained model, and publishes the prediction
back to a status topic.

Includes a minimal HTTP health-check endpoint (Flask, port 7860) so this can
run on Hugging Face Spaces: free Spaces sleep after a period of no incoming
HTTP traffic, so an external uptime pinger (e.g. UptimeRobot, cron-job.org)
hitting this endpoint every few minutes keeps the Space -- and therefore the
MQTT listener -- awake continuously.

NOTE: credentials are hardcoded below since this repo is private. If the
repo's visibility ever changes, rotate the HiveMQ password and update it
here.
"""

import json
import threading
import time
import joblib
import numpy as np
import paho.mqtt.client as mqtt
from flask import Flask, jsonify

MODEL_PATH = "printer_fault_model.joblib"

# --- HiveMQ Cloud credentials ---
MQTT_HOST = "152fa86b7709460f8c860cd2732d9920.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "PrintPulse"
MQTT_PASSWORD = "FinalYearProject@2026"

# --- Topics ---
FEATURES_TOPIC = "printpulse/features"   # ESP32: vibration-derived features
PRINTER_TOPIC = "printpulse/printer"     # PC: nozzle_actual/bed_actual temp readings
STATUS_TOPIC = "printpulse/status"
PRINTER_ID = "printpulse_esp32"  # tag for logs/status payload; no per-device wildcarding needed

# how many consecutive fault predictions in a row before we call it a real
# alert, rather than a single noisy misfire -- tune this once you see live
# behaviour.
CONSECUTIVE_THRESHOLD = 3
_consecutive_fault_count = 0

# Temperature comes from a different device (PC polling the printer) on a
# separate topic than the vibration features (ESP32). We cache the latest
# known reading here and merge it into each incoming feature message, since
# predict() needs both together but they arrive independently.
_latest_temp = {"nozzle_temp": None, "bed_temp": None, "nozzle_target": None, "bed_target": None}
_TEMP_STALE_AFTER_S = 30  # if no temp update in this long, treat as stale
_latest_temp_time = 0

# --- Thermal anomaly thresholds (deterministic rule, not ML) ---
# Overheating: actual exceeds target by more than this margin, sustained.
# This has no "normal ramp-up" ambiguity, since actual only exceeds target
# when something has gone wrong (a properly regulated heater doesn't
# overshoot its setpoint by much).
OVERHEAT_MARGIN_C = 15.0
OVERHEAT_SUSTAIN_S = 10.0

# Underheating / heater failure: actual stays below target by more than this
# margin. Needs a LONGER sustain window than overheating, since a printer
# heating up from cold is expected to sit well below target for a while --
# that's normal startup behaviour, not a fault. A genuine heater/thermistor
# failure looks like this same gap persisting far longer than any real
# heat-up should take.
UNDERHEAT_MARGIN_C = 15.0
UNDERHEAT_SUSTAIN_S = 90.0

# Fallback fixed safety ceiling, used only if the PC-side script isn't
# forwarding nozzle_target/bed_target yet (older payload format). Coarser
# than the actual-vs-target check: it can only catch gross overheating, not
# heater failure, since there's no target to compare against.
FALLBACK_NOZZLE_MAX_C = 260.0
FALLBACK_BED_MAX_C = 100.0

# per-condition "deviation started at" timestamps, so a brief one-off
# reading doesn't trigger an alert -- only a SUSTAINED deviation does
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
    """PC-side temperature readings:
    {"timestamp": ..., "nozzle_actual": ..., "bed_actual": ...,
     "nozzle_target": ..., "bed_target": ...}
    nozzle_target/bed_target are optional -- if your PC script isn't
    forwarding them yet, the thermal check falls back to a fixed safety
    ceiling instead of the more accurate actual-vs-target comparison.
    """
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
    """Returns a thermal anomaly reason string, or None. Tracks sustained
    deviation duration via the module-level _thermal_deviation_start dict."""
    now = time.time()

    if target is not None and target > 0:
        # actual-vs-target comparison -- the accurate path
        if actual > target + OVERHEAT_MARGIN_C:
            _thermal_deviation_start[underheat_key] = None  # clear the other condition
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
        # no target available -- coarser fallback, overheat-only
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
    """Deterministic rule-based check, run alongside (not instead of) the
    ML model. Not learned, not approximate -- this is a plain threshold
    comparison, which is the right tool here since thermal overheating is
    directly measurable and doesn't need a classifier to detect."""
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
    """ESP32 vibration features: merge in the latest cached temp reading before predicting."""
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

    # deterministic thermal check runs independently of the ML model and
    # can fire regardless of what the classifier predicts -- overheating
    # matters even if the vibration signature looks otherwise normal
    thermal = check_thermal_anomaly()
    result.update(thermal)

    # simple debounce: only escalate to an "alert" after N consecutive
    # fault predictions in a row, to avoid reacting to a single noisy window.
    # Thermal anomalies bypass this debounce entirely -- they already
    # require a sustained deviation (10-90s) before check_thermal_anomaly()
    # reports them at all, so no additional debouncing is needed there.
    if result["is_fault"]:
        _consecutive_fault_count += 1
    else:
        _consecutive_fault_count = 0
    result["consecutive_fault_windows"] = _consecutive_fault_count
    result["alert"] = (_consecutive_fault_count >= CONSECUTIVE_THRESHOLD) or thermal["thermal_anomaly"]
    result["printer_id"] = PRINTER_ID

    with _state_lock:
        _service_state["last_prediction"] = result

    client.publish(STATUS_TOPIC, json.dumps(result), qos=1)

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


# --- Flask health endpoint, run in a background thread alongside MQTT ---
app = Flask(__name__)


@app.route("/")
def health():
    with _state_lock:
        state_copy = dict(_service_state)
    state_copy["uptime_seconds"] = round(time.time() - state_copy["started_at"], 1)
    return jsonify(state_copy)


def run_flask():
    # host 0.0.0.0 required for HF Spaces to route external traffic in;
    # port 7860 is HF's expected app_port for Docker Spaces
    app.run(host="0.0.0.0", port=7860)


def run_mqtt():
    client = mqtt.Client(client_id="printer-fault-inference", protocol=mqtt.MQTTv5)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set()  # HiveMQ Cloud requires TLS on port 8883
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message

    client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
    client.loop_forever()  # blocks -- this is why it runs on its own thread


def main():
    mqtt_thread = threading.Thread(target=run_mqtt, daemon=True)
    mqtt_thread.start()
    run_flask()  # blocks in the main thread, which is what HF Spaces expects


if __name__ == "__main__":
    main()

