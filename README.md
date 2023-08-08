# Predicting Self-report items and total scores from digital sensing data
This codebase is intended to analyze data from the OPTIMA study conducted at UCLA. Code used to generate pipeline as part of the Wellcome Leap MCPsych consortium.

## Running models
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
