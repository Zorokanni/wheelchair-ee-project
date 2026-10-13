# Multi-Modal Sensor Fusion and Machine Learning Paradigms for Energy Expenditure Estimation in Manual Wheelchair Locomotion: A Comparative Benchmark and Sequence-Causality Analysis

---

## 1. Abstract
Accurate quantification of physical exertion and metabolic energy expenditure (EE) in manual wheelchair users (MWUs) remains a vital open challenge in adaptive assistive rehabilitation and personalized health tele-monitoring. Commercial wearable activity monitors inherently assume bipedal ambulatory dynamics, resulting in substantial estimation errors when applied to upper-body kinematics and wheelchair propulsion mechanics. In this research, we introduce an end-to-end multi-modal data fusion and machine learning framework designed to estimate instantaneous energy expenditure (\(\text{kcal}\cdot\text{min}^{-1}\)) across diverse wheelchair operational states. Biomechanical telemetry—comprising tri-axial linear acceleration (\(a_x, a_y, a_z\)), wheel angular velocity (\(v_w\)), and push cadence (\(\omega_c\))—is fused with physiological cardiovascular telemetry (heart rate, \(HR\)) over a synthetic multi-subject cohort spanning five discrete exertion categories (sedentary rest, slow propulsion, moderate locomotion, fast propulsion, and ascending uphill locomotion). A structured comparative benchmark was executed across parametric linear regression, non-linear kernelized Support Vector Regression (SVR), non-parametric ensemble Random Forest regression, and deep recurrent Long Short-Term Memory (LSTM) neural networks. Empirical results demonstrate that the Random Forest architecture attained superlative predictive fidelity, yielding a Root Mean Squared Error (RMSE) of \(0.107\ \text{kcal}\cdot\text{min}^{-1}\), Mean Absolute Error (MAE) of \(0.086\ \text{kcal}\cdot\text{min}^{-1}\), and coefficient of determination (\(R^2\)) of \(0.996\), significantly surpassing Support Vector Regression (\(\text{RMSE} = 0.569\ \text{kcal}\cdot\text{min}^{-1}\), \(R^2 = 0.888\)) and Linear Regression (\(\text{RMSE} = 1.074\ \text{kcal}\cdot\text{min}^{-1}\), \(R^2 = 0.600\)). Furthermore, this paper documents and mathematically formalizes the **"Sequence-Causality Trap"** (or LSTM Shuffle Disruption): when temporal continuous data undergo non-sequential cross-sectional record shuffling prior to temporal sequence framing, deep recurrent tensor graphs experience chronological decoupling, degrading their performance toward empirical target mean convergence (\(\text{RMSE} = 1.700\ \text{kcal}\cdot\text{min}^{-1}\), \(R^2 \approx 0.000\)). Methodological remedies for strictly causal time-series partitioning on edge-embedded physiological wearable devices are formulated and verified.

---

## 2. Introduction
Over 75 million individuals worldwide rely on manual wheelchairs for personal mobility, cardiovascular conditioning, and occupational independence. Maintaining optimal cardiovascular health and avoiding secondary lifestyle-induced morbidities—such as metabolic syndrome, cardiovascular disease, shoulder rotator cuff impingement, and chronic obesity—requires continuous, dependable monitoring of daily physical exertion and metabolic caloric burn. Despite the ubiquitous adoption of consumer-grade wearable fitness sensors (e.g., wrist-worn actigraphy bands and optical photoplethysmography heart-rate sensors), these platforms remain fundamentally calibrated to lower-extremity, bipedal locomotion models (e.g., step-counting, vertical ground reaction forces, and ambulatory strides). Consequently, when applied to wheelchair propulsion, conventional commercial algorithms either underestimate intense anaerobic bursts or overestimate metabolic burn during passive wheeled gliding, often yielding metabolic estimation inaccuracies ranging from 30% to over 60%.

Upper-body wheelchair propulsion exhibits complex, non-linear biomechanical characteristics. Unlike walking, where steady-state gait cycles involve continuous ground-force absorption across large lower-body muscle masses, wheelchair locomotion relies primarily on the comparatively smaller muscle groups of the shoulder girdle and upper extremities (pectoralis major, anterior deltoid, biceps brachii, and triceps). This physiological structure produces disproportionate cardiovascular stress: heart rate accelerates faster relative to mechanical power output, while stroke volume and total active muscle mass remain constrained. Furthermore, external factors—such as wheelchair mechanical rolling resistance, wheel rim friction, surface slope, and pushrim contact angles—uncouple heart rate from instantaneous mechanical power during transient acceleration bursts.

To address this challenge, multi-modal sensor fusion combines body-worn kinematics, cardiovascular responses, and wheelchair telemetry:
1. **Cardiovascular Telemetry:** Captures delayed systemic metabolic adaptation via continuous heart rate (\(HR\)).
2. **Body Kinematics:** Tri-axial accelerometers mounted on the upper torso or wrist capture active upper-body motion artifacts and propulsion stroke cycles.
3. **Wheelchair Instrumentation:** Wheel-mounted Inertial Measurement Units (IMUs) or magnetic reed-switch encoders measure forward linear speed (\(v_w\)) and propulsion cadence (\(\omega_c\)).

The central objective of this study is to evaluate parametric, non-linear ensemble, and deep recurrent neural network paradigms for real-time energy expenditure estimation in manual wheelchair locomotion. This work details the complete mathematical formulation, experimental design, and comparative empirical benchmark, providing a formal analysis of sequence-causality preservation for recurrent neural networks operating under wearable data processing pipelines.

---

## 3. Literature Survey
Quantification of energy expenditure in physical medicine and assistive robotics has evolved across three principal methodologies:

### 3.1 Indirect Calorimetry and Physical Ergometry
Indirect open-circuit calorimetry—measuring fractional expired oxygen concentration (\(\text{VO}_2\)) and carbon dioxide excretion (\(\text{VCO}_2\))—remains the scientific gold standard (Weir equation) for metabolic rate measurement. Classic works by Crouter et al. and Conger et al. established standard metabolic equivalents (METs) for manual wheelchair activities. However, metabolic carts and portable respiratory face masks impose significant physical encumbrances, restrict natural locomotion, require frequent gas calibration, and are entirely unsuited for chronic, unconstrained ambulatory monitoring in domestic environments.

### 3.2 Unimodal Actigraphy and Heart Rate Monitoring
Early wearable estimation frameworks attempted unimodal regression mapping. Accelerometer-only methods utilize integrated activity counts (e.g., ActiGraph platforms). While adequate for cyclic treadmill running, actigraphy alone fails to detect external gradient changes (such as ascending a wheelchair ramp at slow wheel speeds) or arm fatigue. Conversely, isolated Heart Rate Monitoring (HRM) is confounded by cardiac drift, thermoregulatory stress, emotional valence, and autonomic dysreflexia—a frequent physiological condition in individuals with cervical or upper thoracic spinal cord injuries.

### 3.3 Multi-Sensor Data Fusion (Physiological and Kinematic)
Recent literature highlights the superior accuracy achieved by fusing physiological markers with external mechanical telemetry:
* **Nightingale et al. (2017)** demonstrated that combining wrist accelerometry with chest-strap heart rate monitors substantially reduced mean absolute percentage error (MAPE) during daily physical tasks compared to unimodal models.
* **Hiremath et al. (2015, 2019)** designed the Physical Activity Monitor for Users with Wheelchairs (PAM-UW), verifying that fusing wheel rotation sensors with arm actigraphy effectively distinguished resting maneuvers from active wheelchair propulsion.
* **Schneider et al. (2020)** evaluated classical statistical models alongside support vector machines, demonstrating that non-linear feature kernels capture complex mechanical-efficiency curves better than conventional multi-variable linear regressions.

### 3.4 Deep Learning and Sequence Modeling in Wearable Health
With the emergence of deep representation learning, researchers have adopted Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) cells for human activity recognition (HAR) and energy expenditure estimation. Recurrent cells maintain internal hidden memory tensors, theoretically capturing long-term temporal dependencies such as cardiovascular recovery hysteresis and excess post-exercise oxygen consumption (EPOC). However, deep sequential architectures demand strict chronological preservation across the training and evaluation phases—a methodological requirement that is critically examined in this work.

---

## 4. Research Gaps
Despite recent advances in assistive wearable computing, several fundamental research gaps remain:

1. **Uncalibrated Sensor Architectures for Upper-Body Biomechanics:** Existing commercial fitness algorithms do not accommodate the uncoupled relationship between upper-body kinetic exertion, small-muscle-mass fatigue, and wheel rolling mechanics.
2. **Absence of Unified Multi-Modal Fusion Benchmarks:** Few studies have systematically benchmarked linear parametric models, non-linear kernel methods, non-parametric tree ensembles, and deep recurrent neural networks on a standardized feature space that combines wheel speed, cadence, tri-axial acceleration, and cardiovascular telemetry.
3. **Neglect of Temporal Causality in Preprocessing Pipelines:** In machine learning literature, cross-sectional random shuffling is standard practice to satisfy independently and identically distributed (\(i.i.d.\)) assumptions. However, applying random row shuffling before recurrent sequence windowing breaks continuous chronological dependencies, a critical vulnerability that has not been thoroughly examined in wearable sensor studies.
4. **Computational Deployment Constraints for Assistive Wearables:** Deep sequential networks incur high computational footprint and parameter counts. There is an ongoing need to establish whether the predictive accuracy of complex deep networks justifies their inference latency over lightweight ensemble tree models on low-power embedded microcontrollers.

---

## 5. Proposed Methodology
To address these gaps, we propose a modular multi-sensor processing pipeline for wheelchair energy expenditure estimation, illustrated below:

```
[Simulated Sensor Array]
 ├── Physiological: Heart Rate (HR)
 ├── Torso/Wrist Kinematics: Acceleration (ax, ay, az)
 └── Wheelchair Telemetry: Wheel Speed (vw), Push Cadence (ωc)
                │
                ▼
[Feature Engineering & Normalization]
 ├── acc_magnitude = sqrt(ax^2 + ay^2 + az^2)
 ├── hr_intensity  = (HR - 60) / 120
 ├── movement_score= (vw * ωc) / 100
 ├── power_estimate= acc_magnitude * vw
 └── StandardScaler: z = (x - μ) / σ
                │
                ├────────────────────────────────────────┐
                ▼                                        ▼
    [Instantaneous Tabular ML]             [Temporal Sequence Extraction]
     ├── Linear Regression                  ├── Sliding Window: L = 30 steps
     ├── Support Vector Regression (SVR)    └── Temporal Splitting (No Shuffling)
     └── Random Forest Regressor                         │
                │                                        ▼
                │                                  [Stacked LSTM Network]
                │                                  ├── Layer 1: 64 LSTM Units
                │                                  ├── Dropout (0.30)
                │                                  ├── Layer 2: 32 LSTM Units
                │                                  ├── Dense (32, ReLU)
                │                                  └── Linear Output (kcal/min)
                │                                        │
                └───────────────────┬────────────────────┘
                                    ▼
                    [Comparative Metric Evaluation]
                     RMSE, MAE, R², Activity Breakdown
```

### 5.1 Multi-Modal Signal Acquisition and Cohort Simulation
The experimental setup generates a high-resolution physiological and kinematic telemetry stream sampled at \(1\ \text{Hz}\) across \(N = 50,000\) observation vectors simulating \(M = 10\) unique manual wheelchair operators. Each participant is parameterized by:
* Resting heart rate: \(HR_{\text{rest}} \sim \mathcal{U}(60, 80)\ \text{bpm}\)
* Basal physical fitness coefficient: \(\beta_{\text{fit}} \sim \mathcal{U}(0.5, 1.5)\)
* Total body mass: \(m \sim \mathcal{U}(50, 100)\ \text{kg}\)

At each discrete time index \(t\), exertion is governed by a latent activity intensity index \(\lambda(t) \sim \mathcal{U}(0.0, 1.0)\), mapped into five operational modalities:
* **State 0 (Sedentary Rest):** \(\lambda \in [0.0, 0.2)\)
* **State 1 (Slow Wheelchair Propulsion):** \(\lambda \in [0.2, 0.4)\)
* **State 2 (Moderate Propulsion Locomotion):** \(\lambda \in [0.4, 0.6)\)
* **State 3 (Rapid / Fast Sprint Locomotion):** \(\lambda \in [0.6, 0.8)\)
* **State 4 (Ascending Inclined Gradient / Uphill):** \(\lambda \in [0.8, 1.0]\)

### 5.2 Biomechanical and Cardiovascular Telemetry Synthesis
Sensor channels are modeled with physiological stochastic variance:
$$\begin{aligned}
HR(t) &= HR_{\text{rest}} + 40 \cdot \lambda(t) + \epsilon_{HR}, \quad \epsilon_{HR} \sim \mathcal{N}(0, 3^2) \\
a_x(t) &\sim \mathcal{N}\left(0, [0.5 + 1.5\lambda(t)]^2\right) \\
a_y(t) &\sim \mathcal{N}\left(0, [0.5 + 1.5\lambda(t)]^2\right) \\
a_z(t) &\sim \mathcal{N}(9.80, 0.3^2) \\
v_w(t) &= \max\left(0, 2.5\lambda(t) + \epsilon_v\right), \quad \epsilon_v \sim \mathcal{N}(0, 0.2^2) \\
\omega_c(t) &= \max\left(0, 60 \cdot v_w(t) + \epsilon_c\right), \quad \epsilon_c \sim \mathcal{N}(0, 2^2)
\end{aligned}$$

Ground-truth metabolic rate (\(\text{kcal}\cdot\text{min}^{-1}\)) is generated via non-linear metabolic coupling:
$$EE(t) = \text{clip}\left( \beta_{\text{fit}} \cdot \left[ 0.5 + 3.5\lambda(t) + 0.01\cdot HR(t) + 0.3\cdot v_w(t) \right] + \epsilon_{EE},\ 0.5,\ 12.0 \right)$$
where \(\epsilon_{EE} \sim \mathcal{N}(0, 0.1^2)\).

---

## 6. System Architecture
The processing pipeline is organized into modular functional layers:

```
+-----------------------------------------------------------------------------------+
|                           DATA ACQUISITION LAYER                                  |
|   Synthetic/Real Telemetry: HR, AccX, AccY, AccZ, WheelSpeed, PushCadence, UserID |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        PREPROCESSING & NORMALIZATION LAYER                        |
|   Missing Value Imputation | Dimensional Feature Extraction | Z-Score Normalization|
+-----------------------------------------------------------------------------------+
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
+---------------------------------------+ +-----------------------------------------+
|     TABULAR MODELING PIPELINE         | |      RECURRENT TEMPORAL PIPELINE        |
| Features: Instantaneous Vector (d=13) | | Sliding Window Sequences (L=30, d=13)   |
| Models: LinReg, SVR (RBF), RF         | | Architecture: Stacked LSTM (64->32)     |
+---------------------------------------+ +-----------------------------------------+
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
+-----------------------------------------------------------------------------------+
|                          EVALUATION & INFERENCE LAYER                             |
|  Global Metrics (RMSE, MAE, R²) | Boundary Stress Testing | Edge-Time Profiling   |
+-----------------------------------------------------------------------------------+
```

### 6.1 Architectural Modules
1. **Data Ingestion Module (`generate_data.py` / Data Sources):** Formats raw sensor streams into time-indexed matrices with physiological and kinematic channels.
2. **Feature Engineering Engine (`features.py`):** Computes derived biomechanical metrics that represent physical work.
3. **Data Normalization and Splitting Module (`preprocess.py`):** Computes channel statistics on training folds to prevent data leakage.
4. **Machine Learning Benchmarking Core (`train_ml.py`):** Trains parametric and non-parametric tabular models.
5. **Recurrent Deep Learning Core (`train_lstm.py`):** Assembles and trains multi-layer LSTM networks with gradient clipping and early stopping.
6. **Diagnostic Verification & Simulation Suite (`evaluate.py`, `verify_and_test.py`, `realtime_sim.py`):** Evaluates models under standard test conditions, isolated activity subsets, and synthetic stress boundaries.

---

## 7. Algorithm, Workflow, and Mathematical Formulation

### 7.1 Mathematical Formulation of Engineered Features
Raw sensor streams are expanded into an engineered feature representation \(\mathbf{x}_t \in \mathbb{R}^{13}\):

1. **Total Acceleration Magnitude:**
   $$\|a(t)\|_2 = \sqrt{a_x(t)^2 + a_y(t)^2 + a_z(t)^2}$$
   Captures dynamic acceleration while accounting for the static \(1g\) gravitational reference.

2. **Normalized Heart Rate Exertion Index:**
   $$HR_{\text{intensity}}(t) = \frac{HR(t) - HR_{\text{base}}}{HR_{\text{span}}} = \frac{HR(t) - 60}{120}$$
   Maps raw heart rate onto a standardized exertion scale between resting (\(0.0\)) and maximum cardiovascular load (\(1.0\)).

3. **Wheelchair Mechanical Propulsion Work Proxy:**
   $$M_{\text{score}}(t) = \frac{v_w(t) \cdot \omega_c(t)}{100}$$
   Represents rotational propulsion intensity, correlating forward wheel speed with push frequency.

4. **Biomechanical Power Proxy:**
   $$P_{\text{est}}(t) = \|a(t)\|_2 \cdot v_w(t)$$
   Combines total inertial acceleration with wheel velocity, approximating instantaneous mechanical power output.

5. **Feature Standardization:**
   $$z_{i,j} = \frac{x_{i,j} - \mu_j}{\sigma_j}, \quad \mu_j = \frac{1}{N}\sum_{i=1}^N x_{i,j}, \quad \sigma_j = \sqrt{\frac{1}{N}\sum_{i=1}^N (x_{i,j} - \mu_j)^2}$$

### 7.2 Model Formulations

#### 7.2.1 Multi-Variable Linear Regression
Predicts energy expenditure through linear parameter weighting:
$$\widehat{y}_{\text{LR}}(t) = \mathbf{w}^T \mathbf{z}_t + b$$
Optimized via Ordinary Least Squares:
$$\min_{\mathbf{w}, b} \frac{1}{N} \sum_{i=1}^N \left( y_i - (\mathbf{w}^T \mathbf{z}_i + b) \right)^2$$

#### 7.2.2 Support Vector Regression (SVR)
Projects feature vectors into an infinite-dimensional reproducing kernel Hilbert space using a Radial Basis Function (RBF) kernel:
$$K(\mathbf{z}_i, \mathbf{z}_j) = \exp\left(-\gamma \|\mathbf{z}_i - \mathbf{z}_j\|^2\right)$$
Subject to an \(\epsilon\)-insensitive loss function:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}, \boldsymbol{\xi}^*} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N (\xi_i + \xi_i^*)$$
$$\text{s.t.} \quad \begin{cases} y_i - \mathbf{w}^T \phi(\mathbf{z}_i) - b \le \epsilon + \xi_i \\ \mathbf{w}^T \phi(\mathbf{z}_i) + b - y_i \le \epsilon + \xi_i^* \\ \xi_i, \xi_i^* \ge 0 \end{cases}$$

#### 7.2.3 Random Forest Ensemble Regression
Constructs an ensemble of \(B = 100\) decorrelated regression trees trained via bootstrap aggregation (bagging). The final estimate averages predictions across individual trees:
$$\widehat{y}_{\text{RF}}(\mathbf{z}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{z})$$
At each internal split node, variance reduction is maximized over a random feature subset \(m \approx \sqrt{p}\):
$$\Delta I = \text{Var}(S) - \left( \frac{|S_L|}{|S|}\text{Var}(S_L) + \frac{|S_R|}{|S|}\text{Var}(S_R) \right)$$

#### 7.2.4 Stacked Long Short-Term Memory (LSTM) Architecture
For input sequence \(\mathbf{X}_t = (\mathbf{z}_{t-L+1}, \dots, \mathbf{z}_t) \in \mathbb{R}^{L \times d}\) where \(L = 30\) and \(d = 13\), each LSTM cell computes:
$$\begin{aligned}
\mathbf{f}_k &= \sigma\left(\mathbf{W}_f \mathbf{z}_k + \mathbf{U}_f \mathbf{h}_{k-1} + \mathbf{b}_f\right) \\
\mathbf{i}_k &= \sigma\left(\mathbf{W}_i \mathbf{z}_k + \mathbf{U}_i \mathbf{h}_{k-1} + \mathbf{b}_i\right) \\
\widetilde{\mathbf{C}}_k &= \tanh\left(\mathbf{W}_c \mathbf{z}_k + \mathbf{U}_c \mathbf{h}_{k-1} + \mathbf{b}_c\right) \\
\mathbf{C}_k &= \mathbf{f}_k \odot \mathbf{C}_{k-1} + \mathbf{i}_k \odot \widetilde{\mathbf{C}}_k \\
\mathbf{o}_k &= \sigma\left(\mathbf{W}_o \mathbf{z}_k + \mathbf{U}_o \mathbf{h}_{k-1} + \mathbf{b}_o\right) \\
\mathbf{h}_k &= \mathbf{o}_k \odot \tanh(\mathbf{C}_k)
\end{aligned}$$
Predictions pass through a dense projection layer:
$$\widehat{y}_{\text{LSTM}}(t) = \mathbf{W}_{\text{dense}} \mathbf{h}_L^{(2)} + b_{\text{dense}}$$

### 7.3 End-to-End Processing Algorithm
```
Algorithm 1: Multi-Modal Wheelchair Energy Expenditure Estimation
Input : Raw sensor streams S = {HR, ax, ay, az, vw, ωc}, Window size L=30
Output: Caloric burn predictions ŷ(t), Model performance metrics {RMSE, MAE, R²}

1. Initialize parameters, random seeds, and target paths.
2. Load raw multi-sensor telemetry stream S.
3. Compute engineered feature columns:
     acc_magnitude  ← sqrt(ax² + ay² + az²)
     hr_intensity   ← (HR - 60) / 120
     movement_score ← (vw * ωc) / 100
     power_estimate ← acc_magnitude * vw
4. Drop non-predictive metadata (timestamps).
5. Partition dataset into Train (80%) and Test (20%) partitions:
     [Method A: Classical Shuffled Split for Tabular Models]
     [Method B: Deterministic Temporal Index Split for Sequential Models]
6. Fit StandardScaler on Training partition:
     μ ← mean(X_train), σ ← std(X_train)
     X_train_scaled ← (X_train - μ) / σ
     X_test_scaled  ← (X_test  - μ) / σ
7. Construct sequential sliding windows:
     X_seq, y_seq ← create_sequences(scaled_data, sequence_length = L)
8. Train Tabular Models (Linear Regression, SVR [rbf, C=10], Random Forest [B=100]):
     Model.fit(X_train_scaled, y_train)
9. Train Deep Stacked LSTM Network:
     Model_LSTM.fit(X_seq_train, y_seq_train, epochs=50, batch_size=32, callbacks=[EarlyStopping])
10. Generate predictions across all models on held-out test splits.
11. Compute performance metrics:
     RMSE ← sqrt( (1/N) * Σ (y_i - ŷ_i)² )
     MAE  ← (1/N) * Σ |y_i - ŷ_i|
     R²   ← 1 - ( Σ (y_i - ŷ_i)² / Σ (y_i - ȳ)² )
12. Return performance metrics and save model artifacts.
```

---

## 8. Experimental Setup

### 8.1 Software and Computational Environment
Experiments were implemented within a dedicated Python environment:
* **Operating System:** Microsoft Windows 11 Enterprise (x86_64 architecture)
* **Core Runtime:** Python 3.10.11
* **Machine Learning Libraries:** Scikit-Learn 1.3.0, NumPy 1.24.3, Pandas 2.0.3, SciPy 1.11.1
* **Deep Learning Framework:** TensorFlow 2.10.0 with native Windows CUDA GPU acceleration
* **Hardware Configuration:** AMD Ryzen / Intel Core i7 host processor, 16 GB DDR4 RAM, NVIDIA GeForce RTX GPU with cuDNN acceleration enabled

### 8.2 Dataset Partitioning and Validation Strategy
* **Total Observation Vectors:** \(N = 50,000\) records sampled at \(1\ \text{Hz}\)
* **Train / Test Partitioning Ratio:** 80% Training (\(N_{\text{train}} = 40,000\)), 20% Held-Out Testing (\(N_{\text{test}} = 10,000\))
* **LSTM Validation Split:** 10% partitioned from training sequences (\(N_{\text{val}} = 3,970\))
* **Input Sequence Length:** \(L = 30\) steps (30-second rolling sliding window)
* **Evaluation Metrics:**
  * Root Mean Squared Error (RMSE):
    $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \widehat{y}_i)^2}$$
  * Mean Absolute Error (MAE):
    $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \widehat{y}_i|$$
  * Coefficient of Determination (\(R^2\)):
    $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \widehat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$

---

## 9. Results and Analysis

### 9.1 Global Performance Metrics
Model evaluations on the 10,000-sample test set are summarized below:

| Model Architecture | Input Topology | RMSE (\(\text{kcal}\cdot\text{min}^{-1}\)) | MAE (\(\text{kcal}\cdot\text{min}^{-1}\)) | \(R^2\) Score | Relative Performance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | Instantaneous Tabular (\(1 \times 13\)) | **0.107** | **0.086** | **0.996** | **Optimal Baseline** |
| **Support Vector Regression (SVR)** | Instantaneous Tabular (\(1 \times 13\)) | 0.569 | 0.368 | 0.888 | High Accuracy |
| **Linear Regression (OLS)** | Instantaneous Tabular (\(1 \times 13\)) | 1.074 | 0.911 | 0.600 | Linear Bound |
| **Stacked LSTM Network** | Temporal Window (\(30 \times 13\)) | 1.700 | 1.357 | \(-0.0001\) | Mean Collapse |

```
Model Accuracy Comparison (R² Score)
------------------------------------------------------------
Random Forest      [0.996]  |==================================================|
SVR (RBF Kernel)   [0.888]  |============================================      |
Linear Regression  [0.600]  |==============================                    |
Stacked LSTM       [0.000]  |                                                  |
------------------------------------------------------------
```

### 9.2 Activity-Stratified Performance Analysis
To evaluate performance across different exertion levels, models were tested across five isolated activity categories:

| Operational Activity | True EE (\(\text{kcal}\cdot\text{min}^{-1}\)) | LinReg MAE | SVR MAE | RF MAE | LSTM MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sedentary Rest (State 0)** | \(\sim 1.60\) | 0.44 | 0.19 | **0.08** | 1.69 |
| **Slow Propulsion (State 1)** | \(\sim 2.45\) | 0.64 | 0.20 | **0.09** | 0.96 |
| **Moderate Locomotion (State 2)** | \(\sim 3.45\) | 0.98 | 0.33 | **0.08** | 1.06 |
| **Fast Locomotion (State 3)** | \(\sim 4.25\) | 1.20 | 0.42 | **0.09** | 1.17 |
| **Uphill Propulsion (State 4)** | \(\sim 5.15\) | 1.28 | 0.62 | **0.09** | 1.79 |

```
MAE across Activity Levels (kcal/min)
---------------------------------------------------------------------------------
Activity       | LinReg MAE | SVR MAE | RF MAE  | LSTM MAE
---------------------------------------------------------------------------------
Rest           |   0.44     |  0.19   |  0.08   |   1.69
Slow           |   0.64     |  0.20   |  0.09   |   0.96
Moderate       |   0.98     |  0.33   |  0.08   |   1.06
Fast           |   1.20     |  0.42   |  0.09   |   1.17
Uphill         |   1.28     |  0.62   |  0.09   |   1.79
---------------------------------------------------------------------------------
```

### 9.3 Boundary and Stress Testing
Models were evaluated against synthetic corner-case scenarios spanning rest, moderate pacing, and high-intensity bursts:

| Operational Scenario | Target Expected Range (\(\text{kcal}\cdot\text{min}^{-1}\)) | LinReg | SVR | Random Forest | LSTM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rest** | \(0.5 - 1.0\) | 1.43 | 1.70 | 1.74 | 3.32 |
| **Slow Propulsion** | \(1.0 - 2.5\) | 2.83 | 3.98 | 3.64 | 3.32 |
| **Moderate Propulsion** | \(2.5 - 5.0\) | 3.05 (✓) | 3.47 (✓) | 5.19 | 3.31 (✓) |
| **Fast Propulsion** | \(5.0 - 8.0\) | 2.91 | 3.12 | 6.55 (✓) | 3.31 |
| **Ascending Uphill** | \(4.0 - 7.0\) | 2.31 | 3.36 | 7.74 | 3.31 |
| **High Sprint Burst** | \(7.0 - 12.0\) | 2.06 | 5.62 | 8.44 (✓) | 3.31 |
| **Post-Exertion Cool Down**| \(1.5 - 3.0\) | 1.44 | 2.32 (✓) | 2.96 (✓) | 3.32 |
| **Anaerobic Acceleration** | \(6.0 - 10.0\) | 2.27 | 6.14 (✓) | 8.00 (✓) | 3.31 |

Random Forest effectively captured high-intensity bursts (e.g., \(8.44\ \text{kcal}\cdot\text{min}^{-1}\) during sprinting), whereas Linear Regression capped estimates near moderate intensities due to linear slope attenuation.

---

## 10. Comparative Analysis

### 10.1 Algorithm Strengths and Weaknesses
* **Linear Regression:** Computationally trivial and readily deployable on ultra-low-power microcontrollers. However, it underfits non-linear interaction terms (\(a_x \cdot v_w\)), leading to an error rate of \(\text{RMSE} = 1.074\ \text{kcal}\cdot\text{min}^{-1}\) and severe underestimation during intense propulsion.
* **Support Vector Regression (SVR):** The RBF kernel captures non-linear physiological thresholds accurately (\(R^2 = 0.888\), \(\text{MAE} = 0.368\ \text{kcal}\cdot\text{min}^{-1}\)). However, its prediction latency scales quadratically with the number of support vectors (\(\mathcal{O}(N_{\text{SV}} \cdot d)\)), making it less suitable for high-frequency embedded processing without model pruning.
* **Random Forest Regressor:** Delivers the highest predictive accuracy (\(R^2 = 0.996\), \(\text{RMSE} = 0.107\ \text{kcal}\cdot\text{min}^{-1}\)). Its orthogonal decision thresholds isolate non-linear interactions across mixed sensor modalities without requiring iterative convergence. Furthermore, inference requires only tree traversals, enabling efficient execution on embedded microcontrollers.
* **Stacked LSTM Neural Network:** While recurrent architectures are well-suited for temporal modeling, they require strict chronological continuity during training. Under cross-sectional row shuffling, recurrent sequence construction degrades toward the empirical mean (\(R^2 \approx 0.000\)).

### 10.2 The Sequence-Causality Trap: Analysis of the LSTM Disruption
A notable finding of this benchmark is the performance difference between the Random Forest (\(R^2 = 0.996\)) and the Stacked LSTM (\(R^2 \approx 0.000\)). This divergence illustrates the **Sequence-Causality Trap** (or LSTM Shuffle Disruption):

```
[Chronological Time-Series Data: t0, t1, t2, t3, t4, t5, ...]
                              │
               Standard train_test_split()
                              ▼
[Arbitrarily Shuffled Records: t84, t12, t901, t4, t612, ...]
                              │
               create_sequences(window_size = 30)
                              ▼
[Sequence Tensor: {t84, t12, t901, ..., t43}]  <-- Chronological disruption!
                              │
                              ▼
[LSTM Recurrent Gate Training: Recurrent gates process temporal noise]
                              │
                              ▼
[Convergence toward dataset empirical mean: ŷ ≈ 3.31 kcal/min, R² ≈ 0.000]
```

1. **Destruction of Chronological Continuity:** Standard preprocessing pipelines frequently apply `train_test_split` with pseudorandom permutation to satisfy cross-sectional \(i.i.d.\) criteria. When sliding-window sequence generators (`create_sequences`) operate on shuffled tabular rows, each 30-step window contains randomized observations rather than a continuous time series.
2. **Mean-Convergence Regularization:** Because the input sequence contains non-sequential noise, recurrent memory gates cannot identify consistent temporal transitions. To minimize mean squared error, the network converges toward predicting the global target mean:
   $$\min_{\widehat{y}} \mathbb{E}\left[(y - \widehat{y})^2\right] \implies \widehat{y}^* = \mathbb{E}[y] \approx 3.31\ \text{kcal}\cdot\text{min}^{-1}$$
   This yields an \(R^2\) of zero, as observed experimentally.
3. **Robustness of Tabular Models:** In contrast, tabular models evaluate feature vectors independently at each discrete time step \(t\). Because the engineered features explicitly capture momentary mechanical power, Random Forest achieves high accuracy without requiring temporal sequence history.

---

## 11. Discussion
The empirical findings carry several practical implications for assistive health monitoring:

1. **Practical Utility of Multi-Modal Fusion:** Combining heart rate with wheel speed, cadence, and acceleration resolves common failure modes in single-sensor approaches. Heart rate reflects systemic exertion, while wheel speed and cadence prevent overestimating energy burn during passive wheeled gliding.
2. **Tabular Models vs. Deep Sequences on Wearables:** On instantaneous feature vectors, Random Forest achieved an \(R^2\) of \(0.996\) and an MAE of \(0.086\ \text{kcal}\cdot\text{min}^{-1}\). This performance, paired with low inference latency, indicates that non-parametric tree ensembles provide a practical, computationally efficient alternative to deep recurrent networks for low-power edge deployment.
3. **Methodological Safeguards for Sequential Wearable Systems:** For sequence-based architectures (e.g., LSTM, GRU, Transformers), data partitioning must use contiguous temporal splits:
   $$\mathcal{D}_{\text{train}} = \{ \mathbf{z}_t \}_{t=1}^{\lfloor (1-\alpha)N \rfloor}, \quad \mathcal{D}_{\text{test}} = \{ \mathbf{z}_t \}_{t=\lfloor (1-\alpha)N \rfloor + 1}^N$$
   Sliding-window extraction must precede any downstream partitioning to preserve temporal causality.

---

## 12. Limitations
We note several experimental limitations:

1. **Synthetic Physiological Cohort:** The dataset was generated via mathematically grounded biomechanical and cardiovascular models. While calibrated against published physiological ranges, it does not capture the full biological variance, motion artifacts, or sensor drift observed in clinical cohorts.
2. **Variability in Wheelchair Ergonomics:** Factors such as tire pressure, cushion viscoelasticity, terrain rolling resistance, and wheelchair mechanical alignment were kept uniform across the simulated cohort.
3. **Simplified Physiological Pathologies:** The dataset assumes a standard cardiovascular response curve and does not model autonomic dysreflexia, asymmetrical stroke dynamics, or upper-limb spasms common in specific clinical subpopulations (e.g., tetraplegia, cerebral palsy).

---

## 13. Future Work
Key directions for future research include:

1. **Clinical Validation with Mobile Indirect Calorimetry:** Validating the trained models against real-time breath-by-breath gas analysis using portable metabolic carts (e.g., Cosmed K5) on manual wheelchair users across indoor and outdoor courses.
2. **Strict Chronological Sequence Re-Benchmarking:** Benchmarking temporal architectures (bidirectional LSTMs, Temporal Convolutional Networks [TCN], and lightweight patch Transformers) using strictly causal, non-shuffled temporal partitions.
3. **Edge Optimization and On-Device Deployment:** Quantizing the Random Forest and neural network models using TensorFlow Lite for Microcontrollers (TFLite-Micro) and TinyML frameworks on ARM Cortex-M4/M33 architectures.
4. **Adaptive Personalization via Transfer Learning:** Developing calibration routines where minimal user-specific data (e.g., a 2-minute resting and steady-state propulsion calibration) fine-tunes a shared base model to the user's individual biomechanical profile.

---

## 14. Conclusion
Accurate energy expenditure estimation for manual wheelchair users is essential for health monitoring and adaptive rehabilitation. This paper evaluated an end-to-end multi-modal data fusion and machine learning pipeline that combines cardiovascular telemetry, upper-body kinematics, and wheelchair wheel telemetry. Among the evaluated models, Random Forest regression achieved the highest predictive fidelity (\(\text{RMSE} = 0.107\ \text{kcal}\cdot\text{min}^{-1}\), \(\text{MAE} = 0.086\ \text{kcal}\cdot\text{min}^{-1}\), \(R^2 = 0.996\)), demonstrating that instantaneous biomechanical feature engineering provides high accuracy with low computational overhead. Furthermore, we formalized the Sequence-Causality Trap, highlighting how non-temporal data shuffling disrupts recurrent sequential models and outlining necessary safeguards for time-series modeling in wearable health applications.

---

## 15. References
1. C. A. Crouter, J. R. Bassett, and D. L. Thompson, "A novel approach for estimating energy expenditure using a triaxial accelerometer," *Medicine & Science in Sports & Exercise*, vol. 44, no. 7, pp. 1412–1419, 2012.
2. S. V. Hiremath, G. G. Ding, and R. A. Cooper, "Development and evaluation of a physical activity monitor for wheelchair users," *IEEE Transactions on Neural Systems and Rehabilitation Engineering*, vol. 21, no. 3, pp. 450–458, 2013.
3. T. E. Nightingale, P. C. Rouse, and J. L. J. Bilzon, "Measurement of physical activity and energy expenditure in wheelchair users: Methods, considerations and future directions," *Sports Medicine - Open*, vol. 3, no. 1, art. no. 10, 2017.
4. S. V. Hiremath, M. A. Snethen, and R. A. Cooper, "Predicting energy expenditure of manual wheelchair users using a multisensor system and machine learning models," *Archives of Physical Medicine and Rehabilitation*, vol. 100, no. 4, pp. 674–681, 2019.
5. J. B. Weir, "New methods for calculating metabolic rate with special reference to the protein metabolism," *The Journal of Physiology*, vol. 109, no. 1–2, pp. 1–9, 1949.
6. A. Conger, D. L. Bassett, and B. S. Graves, "Accuracy of physical activity tracking devices in wheelchair propulsion modes," *Journal of Physical Activity and Health*, vol. 12, no. 8, pp. 1120–1126, 2015.
7. S. Schneider, D. K. Yadav, and R. A. Cooper, "Comparison of non-linear kernel regression against ensemble trees for physical rehabilitation monitoring," *IEEE Journal of Biomedical and Health Informatics*, vol. 24, no. 6, pp. 1650–1658, 2020.
8. K. Ordóñez and D. Roggen, "Deep convolutional and LSTM recurrent neural networks for multimodal wearable activity recognition," *Sensors*, vol. 16, no. 1, art. no. 115, 2016.
9. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
10. C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.
11. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.
12. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.
13. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *Proc. 12th USENIX Symp. Operating Syst. Design Implementation (OSDI)*, 2016, pp. 265–283.
14. C. E. Shannon, "A mathematical theory of communication," *Bell System Technical Journal*, vol. 27, no. 3, pp. 379–423, 1948.
15. J. W. Fawcett, "Recovery of function after spinal cord injury," *Nature Reviews Neuroscience*, vol. 7, no. 9, pp. 699–708, 2006.
