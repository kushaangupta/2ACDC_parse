#!/bin/bash
INPUT_PATH=$1
OUTPUT_PATH=$2
SETTINGS_FILE=$3
SESSION_ID=$4
CUESET=$5
REWARD_ID=$6
TRIAL_STATUS=$7
BIN_SIZE=$8
FORCE_RECALC=$9
CONDA_ENV=${10:-vr2p}  # Default to 'vr2p' if 10th argument is not provided

echo "input path: $INPUT_PATH"
echo "output path: $OUTPUT_PATH"
echo "session id: $SESSION_ID"
echo "cueset: $CUESET"
echo "reward id: $REWARD_ID"
echo "trial status: $TRIAL_STATUS"
echo "bin size: $BIN_SIZE"
echo "force recalc: $FORCE_RECALC"

conda activate $CONDA_ENV
echo $CONDA_DEFAULT_ENV

RESULT=$(python <<EOF
from linear2ac.cluster.placefield import process_placefield_data
process_placefield_data("$INPUT_PATH", "$OUTPUT_PATH", "$SETTINGS_FILE", $SESSION_ID, "$CUESET", $REWARD_ID, "$TRIAL_STATUS",bin_size=$BIN_SIZE,force_recalc=$FORCE_RECALC)
EOF
)
echo $RESULT
