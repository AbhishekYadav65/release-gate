import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


app = FastAPI()


# ---------------------------------------------------------
# STATE
# ---------------------------------------------------------

# Stores:
# runId -> {
#     "fingerprint": "...",
#     "response": {...}
# }
RUNS = {}


# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------

SAFE_INTEGER_MAX = 9007199254740991

TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,3})?"
    r"(?:Z|[+-]\d{2}:\d{2})$"
)

HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------
# BASIC VALIDATION HELPERS
# ---------------------------------------------------------

def is_safe_integer(value: Any) -> bool:
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INTEGER_MAX
    )


def is_non_negative_safe_integer(value: Any) -> bool:
    return is_safe_integer(value)


def is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def is_valid_run_id(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def is_valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False

    if not TIMESTAMP_RE.fullmatch(value):
        return False

    try:
        if value.endswith("Z"):
            datetime.fromisoformat(value[:-1] + "+00:00")
        else:
            datetime.fromisoformat(value)

        return True
    except ValueError:
        return False


def timestamp_to_utc(value: str) -> datetime:
    if value.endswith("Z"):
        dt = datetime.fromisoformat(value[:-1] + "+00:00")
    else:
        dt = datetime.fromisoformat(value)

    return dt.astimezone(timezone.utc)


def utf8_key(value: str) -> bytes:
    return value.encode("utf-8")


# ---------------------------------------------------------
# COMPACT JSON + DIGEST
# ---------------------------------------------------------

def compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def calculate_dataset_digest(
    train_row_ids,
    eval_row_ids,
    feature_names,
) -> str:
    payload = {
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": feature_names,
    }

    return hashlib.sha256(compact_json(payload)).hexdigest()


# ---------------------------------------------------------
# REQUEST FINGERPRINT
# ---------------------------------------------------------

def request_fingerprint(body: dict) -> str:
    """
    Used to determine whether the same runId is being reused
    with the same selection input or a different selection input.
    """
    raw = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------
# ERROR RESPONSE HELPERS
# ---------------------------------------------------------

def selection_error_response(run_id, codes):
    codes = sorted(set(codes), key=utf8_key)

    return {
        "runId": run_id,
        "selectedTrialId": None,
        "trainRowIds": [],
        "evalRowIds": [],
        "featureNames": [],
        "datasetDigest": None,
        "reasonCodes": codes,
    }


def evaluation_error_response(
    run_id,
    selected_trial_id,
    dataset_digest,
    bytes_processed,
    codes,
):
    codes = sorted(set(codes), key=utf8_key)

    return {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": None,
        "criticalSlicePass": False,
        "decision": "reject",
        "bytesProcessed": bytes_processed,
        "reasonCodes": codes,
    }


# ---------------------------------------------------------
# FEATURE VALIDATION
# ---------------------------------------------------------

def validate_feature_object(features):
    if not isinstance(features, dict):
        return False

    for feature_name, feature_data in features.items():
        if not isinstance(feature_name, str):
            return False

        if not isinstance(feature_data, dict):
            return False

        if "value" not in feature_data:
            return False

        if "availableAt" not in feature_data:
            return False

        if not is_valid_timestamp(feature_data["availableAt"]):
            return False

    return True


# ---------------------------------------------------------
# SELECTION ROW VALIDATION
# ---------------------------------------------------------

def validate_selection_row(row):
    if not isinstance(row, dict):
        return False

    required = [
        "id",
        "entity",
        "eventTime",
        "predictionTime",
        "version",
        "split",
        "features",
    ]

    for key in required:
        if key not in row:
            return False

    if not isinstance(row["id"], str):
        return False

    if not isinstance(row["entity"], str):
        return False

    if not is_valid_timestamp(row["eventTime"]):
        return False

    if not is_valid_timestamp(row["predictionTime"]):
        return False

    if not is_safe_integer(row["version"]):
        return False

    if row["split"] not in {"TRAIN", "EVAL"}:
        return False

    if not validate_feature_object(row["features"]):
        return False

    # Every availableAt must be <= predictionTime.
    prediction_time = timestamp_to_utc(row["predictionTime"])

    for feature_data in row["features"].values():
        available_at = timestamp_to_utc(feature_data["availableAt"])

        if available_at > prediction_time:
            return False

    return True


# ---------------------------------------------------------
# DEDUPLICATION
# ---------------------------------------------------------

def deduplicate_rows(rows):
    """
    Deduplicate by:
        [entity, UTC(eventTime)]

    Keep:
        1. highest version
        2. if same version, UTF-8-byte-smallest ID
    """

    retained = {}

    for row in rows:
        key = (
            row["entity"],
            timestamp_to_utc(row["eventTime"]),
        )

        if key not in retained:
            retained[key] = row
            continue

        existing = retained[key]

        if row["version"] > existing["version"]:
            retained[key] = row

        elif row["version"] == existing["version"]:
            if utf8_key(row["id"]) < utf8_key(existing["id"]):
                retained[key] = row

    return list(retained.values())


# ---------------------------------------------------------
# TRIAL VALIDATION
# ---------------------------------------------------------

def validate_trials(trials):
    if not isinstance(trials, list):
        return False, []

    seen_ids = set()

    for trial in trials:
        if not isinstance(trial, dict):
            return False, []

        if "trialId" not in trial:
            return False, []

        if "status" not in trial:
            return False, []

        if "evalMetric" not in trial:
            return False, []

        trial_id = trial["trialId"]

        if not is_safe_integer(trial_id):
            return False, []

        if trial_id in seen_ids:
            return False, []

        seen_ids.add(trial_id)

        if trial["status"] not in {"SUCCEEDED", "FAILED"}:
            return False, []

        if not is_finite_number(trial["evalMetric"]):
            # evalMetric must be finite for the trial to be usable.
            # We don't reject the entire request here; such a trial
            # simply cannot be selected.
            pass

    return True, trials


# ---------------------------------------------------------
# PHASE: SELECT
# ---------------------------------------------------------

def handle_select(body):
    run_id = body.get("runId")

    # Basic input validation
    if not is_valid_run_id(run_id):
        response = selection_error_response(
            run_id if isinstance(run_id, str) else "",
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    required = [
        "phase",
        "runId",
        "forbiddenFeatures",
        "numTrialsLimit",
        "rows",
        "trials",
    ]

    if any(key not in body for key in required):
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    if body["phase"] != "select":
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    if (
        not isinstance(body["forbiddenFeatures"], list)
        or any(not isinstance(x, str) for x in body["forbiddenFeatures"])
    ):
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    if (
        not isinstance(body["numTrialsLimit"], int)
        or isinstance(body["numTrialsLimit"], bool)
        or body["numTrialsLimit"] <= 0
    ):
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    if not isinstance(body["rows"], list) or len(body["rows"]) == 0:
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    if not isinstance(body["trials"], list):
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )
        return JSONResponse(status_code=200, content=response)

    # -----------------------------------------------------
    # RUN ID IDEMPOTENCY / CONFLICT
    # -----------------------------------------------------

    fingerprint = request_fingerprint(body)

    if run_id in RUNS:
        stored = RUNS[run_id]

        if stored["fingerprint"] == fingerprint:
            return JSONResponse(
                status_code=200,
                content=stored["response"],
            )

        return JSONResponse(
            status_code=409,
            content={"error": "RUN_ID_CONFLICT"},
        )

    # -----------------------------------------------------
    # TRIAL LIMIT
    # -----------------------------------------------------

    if len(body["trials"]) > body["numTrialsLimit"]:
        response = selection_error_response(
            run_id,
            ["TRIAL_LIMIT_EXCEEDED"],
        )

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
        }

        return JSONResponse(status_code=200, content=response)

    # -----------------------------------------------------
    # ROW VALIDATION
    # -----------------------------------------------------

    rows = body["rows"]

    row_ids = set()

    for row in rows:
        if not validate_selection_row(row):
            response = selection_error_response(
                run_id,
                ["INVALID_INPUT"],
            )

            RUNS[run_id] = {
                "fingerprint": fingerprint,
                "response": response,
            }

            return JSONResponse(status_code=200, content=response)

        if row["id"] in row_ids:
            response = selection_error_response(
                run_id,
                ["INVALID_INPUT"],
            )

            RUNS[run_id] = {
                "fingerprint": fingerprint,
                "response": response,
            }

            return JSONResponse(status_code=200, content=response)

        row_ids.add(row["id"])

    # -----------------------------------------------------
    # TRIAL VALIDATION
    # -----------------------------------------------------

    trials_valid, trials = validate_trials(body["trials"])

    if not trials_valid:
        response = selection_error_response(
            run_id,
            ["INVALID_INPUT"],
        )

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
        }

        return JSONResponse(status_code=200, content=response)

    # Trial IDs must be unique -- already checked above.

    # -----------------------------------------------------
    # DEDUPLICATE
    # -----------------------------------------------------

    retained_rows = deduplicate_rows(rows)

    # -----------------------------------------------------
    # FIND ELIGIBLE FEATURES
    # -----------------------------------------------------

    forbidden = set(body["forbiddenFeatures"])

    feature_sets = [
        set(row["features"].keys())
        for row in retained_rows
    ]

    common_features = set.intersection(*feature_sets)

    eligible_features = []

    for feature_name in common_features:
        if feature_name in forbidden:
            continue

        eligible = True

        for row in retained_rows:
            available_at = timestamp_to_utc(
                row["features"][feature_name]["availableAt"]
            )

            prediction_time = timestamp_to_utc(
                row["predictionTime"]
            )

            if available_at > prediction_time:
                eligible = False
                break

        if eligible:
            eligible_features.append(feature_name)

    eligible_features.sort(key=utf8_key)

    # -----------------------------------------------------
    # SORT TRAIN/EVAL IDS BY UTF-8
    # -----------------------------------------------------

    train_row_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "TRAIN"
        ],
        key=utf8_key,
    )

    eval_row_ids = sorted(
        [
            row["id"]
            for row in retained_rows
            if row["split"] == "EVAL"
        ],
        key=utf8_key,
    )

    # -----------------------------------------------------
    # SELECT BEST SUCCESSFUL FINITE TRIAL
    # -----------------------------------------------------

    eligible_trials = [
        trial
        for trial in trials
        if (
            trial["status"] == "SUCCEEDED"
            and is_finite_number(trial["evalMetric"])
        )
    ]

    if not eligible_trials:
        response = {
            "runId": run_id,
            "selectedTrialId": None,
            "trainRowIds": train_row_ids,
            "evalRowIds": eval_row_ids,
            "featureNames": eligible_features,
            "datasetDigest": None,
            "reasonCodes": ["NO_SUCCESSFUL_TRIAL"],
        }

        RUNS[run_id] = {
            "fingerprint": fingerprint,
            "response": response,
        }

        return JSONResponse(status_code=200, content=response)

    # Highest metric wins.
    # Exact tie -> smallest trialId.
    best_trial = max(
        eligible_trials,
        key=lambda trial: (
            float(trial["evalMetric"]),
            -trial["trialId"],
        ),
    )

    selected_trial_id = best_trial["trialId"]

    # -----------------------------------------------------
    # DATASET DIGEST
    # -----------------------------------------------------

    dataset_digest = calculate_dataset_digest(
        train_row_ids,
        eval_row_ids,
        eligible_features,
    )

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "trainRowIds": train_row_ids,
        "evalRowIds": eval_row_ids,
        "featureNames": eligible_features,
        "datasetDigest": dataset_digest,
        "reasonCodes": [],
    }

    # -----------------------------------------------------
    # PERSIST COMPLETE RESPONSE
    # -----------------------------------------------------

    RUNS[run_id] = {
        "fingerprint": fingerprint,
        "response": response,
    }

    return JSONResponse(status_code=200, content=response)


# ---------------------------------------------------------
# TEST ROW VALIDATION
# ---------------------------------------------------------

def validate_test_row(row):
    if not isinstance(row, dict):
        return False

    required = [
        "label",
        "prediction",
        "slice",
    ]

    if any(key not in row for key in required):
        return False

    if (
        not isinstance(row["label"], int)
        or isinstance(row["label"], bool)
        or row["label"] not in {0, 1}
    ):
        return False

    if (
        not isinstance(row["prediction"], int)
        or isinstance(row["prediction"], bool)
        or row["prediction"] not in {0, 1}
    ):
        return False

    if not isinstance(row["slice"], str) or row["slice"] == "":
        return False

    return True


# ---------------------------------------------------------
# PHASE: EVALUATE
# ---------------------------------------------------------

def handle_evaluate(body):
    run_id = body.get("runId")
    selected_trial_id = body.get("selectedTrialId")
    dataset_digest = body.get("datasetDigest")
    bytes_processed = body.get("bytesProcessed")

    # -----------------------------------------------------
    # BASIC INPUT VALIDATION
    # -----------------------------------------------------

    basic_valid = True

    if not is_valid_run_id(run_id):
        basic_valid = False

    if not is_safe_integer(selected_trial_id):
        basic_valid = False

    if (
        not isinstance(dataset_digest, str)
        or not HEX64_RE.fullmatch(dataset_digest)
    ):
        basic_valid = False

    if not is_finite_number(body.get("metricFloor")):
        basic_valid = False
    elif not 0 <= float(body["metricFloor"]) <= 1:
        basic_valid = False

    if not isinstance(body.get("requiredSlices"), dict):
        basic_valid = False
    else:
        for name, floor in body["requiredSlices"].items():
            if (
                not isinstance(name, str)
                or not is_finite_number(floor)
                or not 0 <= float(floor) <= 1
            ):
                basic_valid = False
                break

    if not is_non_negative_safe_integer(bytes_processed):
        basic_valid = False

    if not is_non_negative_safe_integer(body.get("maxBytes")):
        basic_valid = False

    if not isinstance(body.get("rows"), list):
        basic_valid = False

    if not basic_valid:
        response = evaluation_error_response(
            run_id if isinstance(run_id, str) else "",
            selected_trial_id if is_safe_integer(selected_trial_id) else None,
            dataset_digest if isinstance(dataset_digest, str) else None,
            bytes_processed if is_non_negative_safe_integer(bytes_processed) else None,
            ["INVALID_INPUT"],
        )

        return JSONResponse(status_code=200, content=response)

    # -----------------------------------------------------
    # LINEAGE CHECK
    # -----------------------------------------------------

    stored = RUNS.get(run_id)

    if stored is None:
        response = evaluation_error_response(
            run_id,
            selected_trial_id,
            dataset_digest,
            bytes_processed,
            ["INVALID_LINEAGE"],
        )

        return JSONResponse(status_code=200, content=response)

    stored_response = stored["response"]

    # A selection is successful only if it has a selected trial
    # and a non-null digest.
    if (
        stored_response.get("selectedTrialId") is None
        or stored_response.get("datasetDigest") is None
        or stored_response.get("selectedTrialId") != selected_trial_id
        or stored_response.get("datasetDigest") != dataset_digest
        or stored_response.get("reasonCodes") != []
    ):
        response = evaluation_error_response(
            run_id,
            selected_trial_id,
            dataset_digest,
            bytes_processed,
            ["INVALID_LINEAGE"],
        )

        return JSONResponse(status_code=200, content=response)

    # -----------------------------------------------------
    # TEST ROW VALIDATION
    # -----------------------------------------------------

    rows = body["rows"]

    if len(rows) == 0:
        # Empty test set:
        # metric becomes null and aggregate/slice checks are skipped.
        response = evaluation_error_response(
            run_id,
            selected_trial_id,
            dataset_digest,
            bytes_processed,
            ["INVALID_TEST_ROW"],
        )

        return JSONResponse(status_code=200, content=response)

    for row in rows:
        if not validate_test_row(row):
            response = evaluation_error_response(
                run_id,
                selected_trial_id,
                dataset_digest,
                bytes_processed,
                ["INVALID_TEST_ROW"],
            )

            return JSONResponse(status_code=200, content=response)

    # -----------------------------------------------------
    # AGGREGATE ACCURACY
    # -----------------------------------------------------

    correct = sum(
        1
        for row in rows
        if row["label"] == row["prediction"]
    )

    test_metric = round(correct / len(rows), 12)

    reason_codes = []

    if test_metric < float(body["metricFloor"]):
        reason_codes.append("AGGREGATE_FLOOR")

    # -----------------------------------------------------
    # SLICE ACCURACY
    # -----------------------------------------------------

    required_slices = body["requiredSlices"]

    critical_slice_pass = True

    for slice_name, floor in required_slices.items():
        slice_rows = [
            row
            for row in rows
            if row["slice"] == slice_name
        ]

        if not slice_rows:
            reason_codes.append(f"MISSING_SLICE:{slice_name}")
            critical_slice_pass = False
            continue

        slice_correct = sum(
            1
            for row in slice_rows
            if row["label"] == row["prediction"]
        )

        slice_accuracy = round(
            slice_correct / len(slice_rows),
            12,
        )

        if slice_accuracy < float(floor):
            reason_codes.append(f"SLICE_FLOOR:{slice_name}")
            critical_slice_pass = False

    # -----------------------------------------------------
    # BYTE LIMIT
    # -----------------------------------------------------

    if bytes_processed > body["maxBytes"]:
        reason_codes.append("BYTE_LIMIT")

    # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    reason_codes = sorted(
        set(reason_codes),
        key=utf8_key,
    )

    decision = "admit" if not reason_codes else "reject"

    # If any required slice failed, criticalSlicePass is false.
    if any(
        code.startswith("MISSING_SLICE:")
        or code.startswith("SLICE_FLOOR:")
        for code in reason_codes
    ):
        critical_slice_pass = False

    response = {
        "runId": run_id,
        "selectedTrialId": selected_trial_id,
        "datasetDigest": dataset_digest,
        "testMetric": test_metric,
        "criticalSlicePass": critical_slice_pass,
        "decision": decision,
        "bytesProcessed": bytes_processed,
        "reasonCodes": reason_codes,
    }

    return JSONResponse(
        status_code=200,
        content=response,
    )


# ---------------------------------------------------------
# POST /bqml
# ---------------------------------------------------------

@app.post("/bqml")
async def bqml(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    if not isinstance(body, dict):
        return JSONResponse(
            status_code=400,
            content={"error": "INVALID_INPUT"},
        )

    phase = body.get("phase")

    if phase == "select":
        return handle_select(body)

    if phase == "evaluate":
        return handle_evaluate(body)

    # Unknown or missing phase:
    # EXACT required response.
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"},
    )


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "bqml",
        "status": "ok",
    }