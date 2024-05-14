set -euo pipefail

# Inputs
SURVEY_DATA=/Users/sakre/Data/OPTIMA/OPTIMA_Surveys/survey_data_2024-04-26_ts.csv
SENSOR_DATA_FOLDER=~/Data/OPTIMA/OPTIMA_HealthKit/HealthKit_Datastreams
SURVEYS="psqi pvss phq14"
THRESHOLDS=~/Data/OPTIMA/PHQ_PVSS_PSQI_binary_thresholds-Jan212024.xlsx

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