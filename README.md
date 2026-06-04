# AUT-2607 Task 4

## Overview

This project investigates machine learning classification of industrial robot tasks using robot operational data and network communication features.

The goal is to predict the robot task type based on robot measurements and network conditions.

Methods used:

* Pearson Correlation
* Cross Correlation
* Mutual Information
* Linear Discriminant Analysis (LDA)
* Multi-Layer Perceptron (MLP)

Evaluation:

* Accuracy
* Confusion Matrix

---

## Dataset

The dataset contains 1000 observations and 25 variables.

Target variable:

```text
task_type
```

Classes:

* Assembly
* Pick-and-Place
* Welding

Features include robot measurements, sensor information and network communication metrics such as latency, packet loss, signal strength and bandwidth allocation.

---

## Preprocessing

The following preprocessing steps were performed:

* Removed `robot_id`
* Removed `position_coordinates`
* Converted `timestamp` into minute-of-hour
* Applied one-hot encoding to categorical variables
* Scaled numerical features using StandardScaler

---

## Results

### LDA

Accuracy:

```text
0.36
```

### MLP

Accuracy:

```text
0.325
```

Both models achieved relatively poor performance and struggled to distinguish between the three task categories.

---

## Correlation Analysis

Pearson correlation showed very weak relationships between the input features and the target variable.

Mutual Information identified a few slightly informative features, including:

* resource_allocation
* network_load
* command_delay

However, the information content remained low overall.

Cross-correlation analysis also showed generally weak relationships between most features.

---

## Conclusion

The selected robot and network features contained limited information about the robot task type.

This was reflected in both the correlation analysis and the machine learning results, where LDA and MLP achieved only modest classification performance.

The dataset therefore appears challenging for the tested machine learning methods.
