from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone
import math
import re

app = FastAPI()

SAFE_INTEGER_MAX = 9007199254740991

VERSION_RE = re.compile(r"^[1-9][0-9]*$")

current_champion = None


def invalid_input():
    return JSONResponse(
        status_code=400,
        content={"error": "INVALID_INPUT"}
    )


def parse_timestamp(value):
    if not isinstance(value, str):
        raise ValueError()

    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$"

    if not re.fullmatch(pattern, value):
        raise ValueError()

    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value

    dt = datetime.fromisoformat(normalized)

    if dt.tzinfo is None:
        raise ValueError()

    return dt


def valid_version(value):
    if not isinstance(value, str):
        return False

    if not VERSION_RE.fullmatch(value):
        return False

    try:
        number = int(value)
    except Exception:
        return False

    return 1 <= number <= SAFE_INTEGER_MAX


def finite_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def safe_integer(value):
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and 0 <= value <= SAFE_INTEGER_MAX
    )


def add_failure(failures, code):
    if code not in failures:
        failures.append(code)


def validate_policy(policy):
    if not isinstance(policy, dict):
        return False

    required = [
        "datasetDigest",
        "schemaDigest",
        "maxAgeSeconds",
        "accuracyFloor",
        "requiredSlices",
        "maxLatencyMs",
        "maxSizeBytes",
        "minImprovement",
    ]

    if any(key not in policy for key in required):
        return False

    if not isinstance(policy["datasetDigest"], str) or not policy["datasetDigest"]:
        return False

    if not isinstance(policy["schemaDigest"], str) or not policy["schemaDigest"]:
        return False

    if not safe_integer(policy["maxAgeSeconds"]):
        return False

    if not finite_number(policy["accuracyFloor"]):
        return False

    if not 0 <= float(policy["accuracyFloor"]) <= 1:
        return False

    if not isinstance(policy["requiredSlices"], dict):
        return False

    for value in policy["requiredSlices"].values():
        if not finite_number(value) or not 0 <= float(value) <= 1:
            return False

    if not finite_number(policy["maxLatencyMs"]):
        return False

    if float(policy["maxLatencyMs"]) < 0:
        return False

    if not safe_integer(policy["maxSizeBytes"]):
        return False

    if not finite_number(policy["minImprovement"]):
        return False

    if not 0 <= float(policy["minImprovement"]) <= 1:
        return False

    return True


def evaluate_version(version, policy, as_of):
    failures = []

    if not isinstance(version, dict):
        return ["INVALID_VERSION"]

    version_id = version.get("version")

    if not valid_version(version_id):
        add_failure(failures, "INVALID_VERSION")

    artifact_digest = version.get("artifactDigest")

    evaluation = version.get("evaluation")

    if evaluation is None or not isinstance(evaluation, dict):
        add_failure(failures, "MISSING_EVALUATION")
        return sorted(set(failures))

    created_at_raw = evaluation.get("createdAt")

    try:
        created_at = parse_timestamp(created_at_raw)
    except Exception:
        add_failure(failures, "INVALID_TIMESTAMP")
        created_at = None

    if created_at is not None:
        if created_at > as_of:
            add_failure(failures, "FUTURE_EVALUATION")
        elif created_at < as_of.timestamp() and False:
            pass

        age = (as_of - created_at).total_seconds()

        if age > policy["maxAgeSeconds"]:
            add_failure(failures, "STALE_EVALUATION")

    accuracy = evaluation.get("accuracy")
    latency = evaluation.get("latencyMs")
    size = evaluation.get("sizeBytes")

    if not finite_number(accuracy):
        add_failure(failures, "NON_FINITE")

    if not finite_number(latency):
        add_failure(failures, "NON_FINITE")

    if not safe_integer(size):
        add_failure(failures, "NON_FINITE")

    if finite_number(accuracy) and not 0 <= float(accuracy) <= 1:
        add_failure(failures, "METRIC_RANGE")

    if finite_number(latency) and float(latency) < 0:
        add_failure(failures, "METRIC_RANGE")

    if isinstance(size, int) and size < 0:
        add_failure(failures, "METRIC_RANGE")

    if evaluation.get("artifactDigest") != artifact_digest:
        add_failure(failures, "ARTIFACT_MISMATCH")

    if evaluation.get("datasetDigest") != policy["datasetDigest"]:
        add_failure(failures, "DATASET_MISMATCH")

    if evaluation.get("schemaDigest") != policy["schemaDigest"]:
        add_failure(failures, "SCHEMA_MISMATCH")

    if finite_number(accuracy):
        if not 0 <= float(accuracy) <= 1:
            pass
        elif float(accuracy) < float(policy["accuracyFloor"]):
            add_failure(failures, "ACCURACY_FLOOR")

    if finite_number(latency) and float(latency) >= 0:
        if float(latency) > float(policy["maxLatencyMs"]):
            add_failure(failures, "LATENCY_LIMIT")

    if isinstance(size, int) and 0 <= size <= SAFE_INTEGER_MAX:
        if size > policy["maxSizeBytes"]:
            add_failure(failures, "SIZE_LIMIT")

    slices = evaluation.get("slices")

    if not isinstance(slices, dict):
        slices = {}

    for name, floor in policy["requiredSlices"].items():
        if name not in slices:
            add_failure(failures, f"MISSING_SLICE:{name}")
            continue

        value = slices[name]

        if not finite_number(value):
            add_failure(failures, f"SLICE_RANGE:{name}")
            continue

        if not 0 <= float(value) <= 1:
            add_failure(failures, f"SLICE_RANGE:{name}")
            continue

        if float(value) < float(floor):
            add_failure(failures, f"SLICE_FLOOR:{name}")

    return sorted(set(failures))


@app.post("/promote")
async def promote(request: Request):
    global current_champion

    try:
        body = await request.json()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    as_of_raw = body.get("asOf")
    champion_version = body.get("championVersion")
    policy = body.get("policy")
    versions = body.get("versions")

    if not isinstance(as_of_raw, str):
        return invalid_input()

    try:
        as_of = parse_timestamp(as_of_raw)
    except Exception:
        return invalid_input()

    if not isinstance(champion_version, str):
        return invalid_input()

    if not valid_version(champion_version):
        return invalid_input()

    if not validate_policy(policy):
        return invalid_input()

    if not isinstance(versions, list):
        return invalid_input()

    seen = set()

    for version in versions:
        if not isinstance(version, dict):
            return invalid_input()

        version_id = version.get("version")

        if not valid_version(version_id):
            return invalid_input()

        if version_id in seen:
            return invalid_input()

        seen.add(version_id)

    lookup = {v["version"]: v for v in versions}

    if champion_version not in lookup:
        return invalid_input()

    failed_gates = {}
    eligible = []

    for version in versions:
        version_id = version["version"]

        failures = evaluate_version(
            version,
            policy,
            as_of
        )

        if failures:
            failed_gates[version_id] = failures
        else:
            eligible.append(version)

    eligible.sort(
        key=lambda v: (
            -float(v["evaluation"]["accuracy"]),
            float(v["evaluation"]["latencyMs"]),
            v["evaluation"]["sizeBytes"],
            int(v["version"])
        )
    )

    eligible_versions = [v["version"] for v in eligible]

    champion = lookup[champion_version]

    champion_failures = failed_gates.get(champion_version, [])

    if champion_failures:
        return {
            "action": "block",
            "championVersion": champion_version,
            "selectedVersion": None,
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": None,
            "evidence": None
        }

    champion_evaluation = champion["evaluation"]

    if current_champion is not None:
        champion_version = current_champion

        if champion_version in lookup:
            champion = lookup[champion_version]
            champion_evaluation = champion["evaluation"]

    best = eligible[0] if eligible else champion

    improvement = round(
        float(best["evaluation"]["accuracy"])
        - float(champion_evaluation["accuracy"]),
        12
    )

    if (
        best["version"] != champion_version
        and improvement >= float(policy["minImprovement"])
    ):
        current_champion = best["version"]

        return {
            "action": "promote",
            "championVersion": champion_version,
            "selectedVersion": best["version"],
            "eligibleVersions": eligible_versions,
            "failedGates": failed_gates,
            "aliasMutation": {
                "alias": "champion",
                "version": best["version"]
            },
            "evidence": best["evaluation"]
        }

    current_champion = champion_version

    return {
        "action": "retain",
        "championVersion": champion_version,
        "selectedVersion": champion_version,
        "eligibleVersions": eligible_versions,
        "failedGates": failed_gates,
        "aliasMutation": None,
        "evidence": champion_evaluation
    }