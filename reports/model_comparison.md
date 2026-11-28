# Model Comparison Report

**Champion Model:** `DecisionTree`

| model_name       |   cv_train_accuracy |   cv_val_accuracy |   gap_pct |   training_time_seconds |   prediction_time_ms |   model_size_mb |
|:-----------------|--------------------:|------------------:|----------:|------------------------:|---------------------:|----------------:|
| GradientBoosting |              0.9681 |            0.9606 |      0.75 |                 36.4345 |              14.2888 |          1.6302 |
| RandomForest     |              0.9634 |            0.956  |      0.74 |                 16.8635 |             124.703  |         29.0696 |
| DecisionTree     |              0.9694 |            0.9518 |      1.75 |                  2.0746 |              12.5148 |          0.1464 |
| KNN              |              0.9198 |            0.9115 |      0.83 |                  0.1442 |              12.9579 |          3.2107 |
| SVM              |              0.874  |            0.8733 |      0.06 |                  4.5623 |              14.4814 |          0.9075 |
| GaussianNB       |              0.8464 |            0.8462 |      0.02 |                  0.9684 |              15.2728 |          0.0072 |
