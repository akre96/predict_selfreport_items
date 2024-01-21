set -euo pipefail

# Inputs
SURVEY_DATA=~/Data/OPTIMA/OPTIMA_Surveys/survey_data_12Oct2023_ts.csv
SENSOR_DATA_FOLDER=~/Data/OPTIMA/OPTIMA_HealthKit/HealthKit_Datastreams
SURVEYS="psqi phq14 pvss"
THRESHOLDS=~/Data/OPTIMA/OPTIMA_survey_binarize-July312023.xlsx

# Outputs
SENSOR_OUTPUT=sensordata.csv
QC_SENSOR_OUTPUT=sensordata_qc.csv
MODEL_OUTPUT=tmp_out

## Predict Self-Report Items
poetry run python predict_selfreport_items/predict_selfreport_evaluation.py \
    --survey-data $SURVEY_DATA \
    --sensor-features $QC_SENSOR_OUTPUT \
    --binary-thresholds $THRESHOLDS \
    --output-folder $MODEL_OUTPUT \
    --debug
