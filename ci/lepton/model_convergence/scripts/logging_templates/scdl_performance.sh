#!/usr/bin/env bash
set -euo pipefail

# Get job name
JOB_NAME="${LEPTON_JOB_NAME:-unknown-job}"
# Run the script
set +e
(
__SCRIPT__
)
RC=$?
set -e

echo "=========================================="
echo "Job completed with exit code: $RC"
echo "=========================================="

# Log contents of current directory
echo ""
echo "=========================================="
echo "Contents of working directory:"
echo "=========================================="
ls -lah
echo ""

# Check for benchmark.json
if [ ! -f "benchmark.json" ]; then
    echo "Warning: benchmark.json not found in current directory"
    exit "$RC"
fi

echo "Found benchmark.json, contents:"
cat benchmark.json | jq '.' 2>/dev/null || cat benchmark.json
echo ""

# Prepare config JSON
ALL_CONFIG_JSON='__ALL_CONFIG_JSON__'
if echo "$ALL_CONFIG_JSON" | jq -e . >/dev/null 2>&1; then
  ALL_CONFIG_JSON_UPDATED="$(printf '%s' "$ALL_CONFIG_JSON" | jq -c '.')"
else
  echo "Warning: ALL_CONFIG_JSON is not valid JSON. Using empty object."
  ALL_CONFIG_JSON_UPDATED='{}'
fi

# Authenticate to Lepton
pip install -q leptonai >/dev/null 2>&1 || pip install -q leptonai || true
lep login -c "$LEP_LOGIN_CREDENTIALS" || true

# Get lepton job details
JOB_INFO="$(
  lep job get --id "$JOB_NAME" 2>/dev/null \
  | awk '
    BEGIN { json=""; depth=0; started=0 }
    {
      for (i=1; i<=length($0); i++) {
        ch = substr($0, i, 1)
        if (ch == "{") { depth++; started=1 }
        if (started)      json = json ch
        if (ch == "}") {
          depth--
          if (started && depth == 0) { print json; exit }
        }
      }
    }' \
  | jq -c '
    {
      metadata: {
        id: .metadata.id,
        name: .metadata.name,
        created_at: .metadata.created_at,
        created_by: .metadata.created_by
      },
      spec: {
        resource_shape: .spec.resource_shape,
        container_image: .spec.container.image,
        completions: .spec.completions,
        parallelism: .spec.parallelism
      },
      status: {
        job_name: .status.job_name,
        state: .status.state,
        ready: .status.ready,
        active: .status.active,
        failed: .status.failed,
        succeeded: .status.succeeded,
        creation_time: .status.creation_time,
        completion_time: .status.completion_time
      }
    }
  ' 2>/dev/null
)"
JOB_INFO_JSON="$(printf '%s' "$JOB_INFO" | jq -c . 2>/dev/null || echo '{}')"

# Read benchmark results
BENCHMARK_JSON=$(cat benchmark.json 2>/dev/null || echo '{}')

# Combine all data
COMBINED_JSON=$(jq -n \
    --argjson job_info "$JOB_INFO_JSON" \
    --argjson config "$ALL_CONFIG_JSON_UPDATED" \
    --argjson benchmark "$BENCHMARK_JSON" \
    '
    {
      job_name: env.LEPTON_JOB_NAME,
      job_info: $job_info,
      config: $config,
      benchmark_results: $benchmark
    }
    ')

echo "$COMBINED_JSON" > data-perf-combined.json
echo "Created combined data file: data-perf-combined.json"

# Upload to Kratos
echo ""
echo "=========================================="
echo "Uploading data performance metrics to Kratos..."
echo "=========================================="

UUID=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "$(date +%s)-$$-$RANDOM")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%S.%3NZ")

if [ -z "${KRATOS_SSA_CLIENT_ID:-}" ] || [ -z "${KRATOS_SSA_SECRET:-}" ] || [ -z "${KRATOS_SSA_URL:-}" ]; then
    echo "Warning: Kratos credentials not found. Skipping telemetry upload."
else
    ENCODED_CREDS=$(echo -n "${KRATOS_SSA_CLIENT_ID}:${KRATOS_SSA_SECRET}" | base64 | tr -d '\n')
    TOKEN_RESPONSE=$(curl -sS --request POST \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -H "Authorization: Basic $ENCODED_CREDS" \
        "https://${KRATOS_SSA_URL}/token?grant_type=client_credentials&scope=telemetry-write" 2>&1)
    ACCESS_TOKEN=$(echo "$TOKEN_RESPONSE" | jq -r '.access_token' 2>/dev/null)

    if [ -n "$ACCESS_TOKEN" ] && [ "$ACCESS_TOKEN" != "null" ]; then
        JSON_PAYLOAD=$(jq -n \
            --arg id "$UUID" \
            --arg time "$TIMESTAMP" \
            --arg source "bionemo-data-performance" \
            --arg type "scdl-benchmark-metrics" \
            --arg subject "scdl_performance" \
            --argjson data "$COMBINED_JSON" \
            '{
              "specversion": "1.0",
              "id": $id,
              "time": $time,
              "source": $source,
              "type": $type,
              "subject": $subject,
              "data": $data
            }')

        RESPONSE=$(curl -sS --request POST \
            -H "Content-Type: application/cloudevents+json" \
            -H "Authorization: Bearer ${ACCESS_TOKEN}" \
            "https://prod.analytics.nvidiagrid.net/api/v2/topic/bionemo-convergence-lepton-logs-kratos.telemetry.lepton-poc-v001.prod" \
            --data "$JSON_PAYLOAD" 2>&1)

        if [ $? -eq 0 ]; then
            echo "✓ Event sent successfully to Kratos (ID: $UUID)"
        else
            echo "Failed to send event to Kratos: $RESPONSE"
        fi
    else
        echo "Error: Failed to get Kratos access token"
    fi
fi

exit "$RC"