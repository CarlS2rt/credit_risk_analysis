# Credit Risk Analysis

## Project Overview

The purpose of this project is to determine the effectiveness of supervised machine learning at predicting credit risk. Six different machine learning models were employed in the project to determine if one (if any) of the models were most effective at predicting credit risk to evaluate if credit should be offered.

## Analysis Results

### Naive Random Oversampling:

```python
Oversampling Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | :------------------ | :----------------- |
| Actual High Risk | 52                  | 22                 |
| Actual Low Risk  | 5852                | 11279              |

```python
Accuracy Score : 0.6805498803338976
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.70      0.66      0.02      0.68      0.46        74
   low_risk       1.00      0.66      0.70      0.79      0.68      0.46     17131

avg / total       0.99      0.66      0.70      0.79      0.68      0.46     17205
```

### SMOTE Oversampling:

```python
SMOTE Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | :------------------ | :----------------- |
| Actual High Risk | 53                  | 21                 |
| Actual Low Risk  | 6371                | 10760              |

```python
Accuracy Score : 0.6721586597396533
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.72      0.63      0.02      0.67      0.45        74
   low_risk       1.00      0.63      0.72      0.77      0.67      0.45     17131

avg / total       0.99      0.63      0.72      0.77      0.67      0.45     17205
```

### Undersampling:

```python
Undersampling Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | :------------------ | :----------------- |
| Actual High Risk | 45                  | 29                 |
| Actual Low Risk  | 9567                | 7564               |

```python
Accuracy Score : 0.5248234195318429
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.00      0.61      0.44      0.01      0.52      0.27        74
   low_risk       1.00      0.44      0.61      0.61      0.52      0.26     17131

avg / total       0.99      0.44      0.61      0.61      0.52      0.26     17205
```

### SMOTEENN Sampling:

```python
SMOTEENN Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | :------------------ | :----------------- |
| Actual High Risk | 56                  | 18                 |
| Actual Low Risk  | 6775                | 10356              |

```python
Accuracy Score : 0.6806374408966203
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.01      0.76      0.60      0.02      0.68      0.46        74
   low_risk       1.00      0.60      0.76      0.75      0.68      0.45     17131

avg / total       0.99      0.61      0.76      0.75      0.68      0.45     17205
```

### Balanced Random Forest Ensemble:

```python
Balanced Random Forest Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | ------------------: | -----------------: |
| Actual High Risk |                  22 |                 52 |
| Actual Low Risk  |                  17 |              17114 |

```python
Accuracy Score : 0.6481524721265542
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.56      0.30      1.00      0.39      0.54      0.28        74
   low_risk       1.00      1.00      0.30      1.00      0.54      0.32     17131

avg / total       1.00      1.00      0.30      1.00      0.54      0.32     17205
```

### Easy Ensemble AdaBoost:

```python
Easy Ensemble AdaBoost Analysis
Confusion Matrix
```

|                  | Predicted High Risk | Predicted Low Risk |
| :--------------- | ------------------: | -----------------: |
| Actual High Risk |                  67 |                  7 |
| Actual Low Risk  |                1018 |              16113 |

```python
Accuracy Score : 0.9229904850855175
Classification Report
                   pre       rec       spe        f1       geo       iba       sup

  high_risk       0.06      0.91      0.94      0.12      0.92      0.85        74
   low_risk       1.00      0.94      0.91      0.97      0.92      0.85     17131

avg / total       1.00      0.94      0.91      0.97      0.92      0.85     17205
```

## Summary

Overall, the supervised machine learning models were only marginally accurate. Five of the six models had an accuracy score under 0.70. The Easy Ensemble AdaBoost method, though, had an accuracy score of 0.92, which indicates the model was highly effective at predicting credit risk. Based on the classification report results of all the models tested, the Easy Ensemble method is recommended due to its high accuracy.