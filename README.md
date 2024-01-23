# Predicting Self-report items and total scores from digital sensing data
This codebase is intended to analyze data from the OPTIMA study conducted at UCLA. Code used to generate pipeline as part of the Wellcome Leap MCPsych consortium.


## Aggregating Apple Watch features

Initial aggergation is handled by the script `predict_selfreport_items/collect_sensor_features.py`. 

**Sensor Data:** This file assumes HealthKit data is stored in a directory structure like where words in brackets are filler words
```
[HEALTHKIT_FOLDER]
├──[HEALTHKIT_DATA_TYPE]
│   └── [USER_ID]-[HEALTHKIT_DATA_TYPE].csv
```

**Survey Data:** This script will also require survey data used to find timestamps that passive data will be aggregated relative to. The CSV is expected to have the following columns:
- user_id: unique identifier for a participant
- response: question response
- survey_start: timestamp of survey response
- redcap_event_name: name of event assocaited with response
- question: Question being asked (reply in response column)
- survey: Name of survey question belongs to
- duration: Timespan a question asks about. For example the PSQI asks about the last month, so duration would be 28days

**Thresholds**: This refers to an excel sheet listing for each question, what value should be use to convert it to a binary response. Requires the columns:
- survey: same as survey data
- question: Same as survey data
- threshold: Value where ≥ this value is set to True


### Example script call
```
# Inputs
SURVEY_DATA=/survey_data.csv
SENSOR_DATA_FOLDER=/HealthKit_Datastreams
SURVEYS="psqi pvss phq14"
THRESHOLDS=PHQ_PVSS_PSQI_binary_thresholds-Jan212024.xlsx

# Outputs
SENSOR_OUTPUT=sensordata.csv
QC_SENSOR_OUTPUT=sensordata_qc.csv


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
```

## Training models

If feature aggregation (prior step) ran, model inputs should be ready to train and evaluate question-level classification of survey responses. This step outputs:
- `[out_dir]/[survey]_[question]/[survey]_[question]_[model].pkl`: Pickled trained model. `[model]` refers to either dummy, gradient boosted, logistic regression, or random forest. 
- `[out_dir]/[survey]_[question]/[survey]_[question]_[model]_predictions.parquet`: Includes model predictions, true values, and SHAP values (for feature importance)
- `[out_dir]/binary_predictions.parquet`: Concatenation of above for all models

### Example script call
```
MODEL_OUTPUT=tmp_out
poetry run python predict_selfreport_items/predict_selfreport_CV.py \
    --survey-data $SURVEY_DATA \
    --sensor-features $QC_SENSOR_OUTPUT \
    --binary-thresholds $THRESHOLDS \
    --output-folder $MODEL_OUTPUT \
    --parallel
```

## Evaluate models
This script takes the model predictions and runs a bootstrapping method that accounts for repeated samples per participant in generating AUC ROC scores and other performance metrics. The best performing models per classification task (question response classification) are evaluated for having ROC AUC scores > 0.5 across folds.

- outputs file with statistical test for significance and multiple testing correction `[out_dir]/binary_performance_test.csv`

### Example script call
```
poetry run python predict_selfreport_items/evaluate_predictions.py \
    --output_folder $MODEL_OUTPUT \
    --predictions_file $MODEL_OUTPUT/binary_predictions.parquet
```


## Running trained models
Models generated from the pipeline can be used to detect self-report features from sensor data.

Example snippet assuming pipeline has been run already with ouputs saved to `tmp_out` directory
```
poetry run python predict_selfreport_items/run_model.py \
    --sensor-features sensordata_qc.csv \
    --model-file tmp_out/phq14_phq14_total_score_model.pkl \
    --output-file phq_14_total_prediction.csv
```

- `sensordata_qc.csv` is the output of collect_sensor_features.py and sensor_feature_qc.py in the predict_selfreport_items folder.
- Full pipeline can be run from the UDCP. It is named "selfreport_item_classification" in the project "ucla_download_test".


## Docker
`Dockerfile` is the source for the image built to akre96/predict_selfreport_items:latest
