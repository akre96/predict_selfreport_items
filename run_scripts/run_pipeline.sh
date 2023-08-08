set -euo pipefail

# Inputs
SURVEY_DATA=~/Data/OPTIMA/survey_data_Aug032023.csv
SENSOR_DATA_FOLDER=~/Data/OPTIMA/HealthKit_Datastreams
SURVEYS="psqi phq14 pvss"
THRESHOLDS=~/Data/OPTIMA/OPTIMA_survey_binarize-July312023.xlsx

# Outputs
SENSOR_OUTPUT=sensordata.csv
QC_SENSOR_OUTPUT=sensordata_qc.csv
MODEL_OUTPUT=tmp_out

## Sensor Feature Extraction
if [ -f $SENSOR_OUTPUT ]; then
    echo "Sensor features already extracted"
else
    poetry run python predict_selfreport_items/collect_sensor_features.py \
        --survey-data $SURVEY_DATA \
        --sensor-data-folder $SENSOR_DATA_FOLDER \
        --output-file $SENSOR_OUTPUT \
        --surveys $SURVEYS \
        --parallel
fi


## Sensor Feature QC
poetry run python predict_selfreport_items/sensor_feature_qc.py \
    --sensor-features $SENSOR_OUTPUT \
    --output-file $QC_SENSOR_OUTPUT \

## Predict Self-Report Items
poetry run python predict_selfreport_items/predict_selfreport_evaluation.py \
    --survey-data $SURVEY_DATA \
    --sensor-features $QC_SENSOR_OUTPUT \
    --binary-thresholds $THRESHOLDS \
    --output-folder $MODEL_OUTPUT \
