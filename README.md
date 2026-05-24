# ReturnRack

**App for sorting the returns received based on the warehouse**

## Overview

_ReturnRack_ is a smart, browser-based returns routing application designed to classify returned parcels to their appropriate warehouse using data-driven AI. By leveraging layered pattern recognition, machine learning, and advanced rule-based logic, it automates return processing across multiple facilities.

This is an end-to-end intelligence dashboard combining user-friendly interfaces with high-performance prediction models, enhancing traceability and sorting efficiency in warehouse operations.

---

## Features

- 📦 **Automated Return Sorting**: AI-powered prediction of correct warehouse for every incoming return parcel.
- 🏭 **Multi-Warehouse & Merchant Ready**: Simultaneous visibility on POZ/BER3 warehouses and arbitrary merchant settings.
- 🔎 **Real-Time Tracking**: Instantly predict warehouse based on tracking code entry or scanning.
- 📈 **Confidence Metrics**: Statistical display of prediction certainty, recent scans, and operational breakdown.
- 🌗 **Modern Responsive UI**: Fast, accessible, with light/dark modes and fully client-side privacy.

---

## Modelling & Algorithms

### Model Overview & Workflow

The ReturnRack core routing engine is structured as a cascaded multi-layered predictor:

1. **Layer 1: Deterministic Rule Engine**
   - Applies hardcoded and dynamically learned prefix/suffix/pattern rules.
   - Example: Given a 20-digit code, specific 4-digit substrings are mapped deterministically to warehouse BER3 with >98% accuracy.
   - Mathematical summary:\
     For input \( x \), the rule engine is a function \( f_{\text{rules}}(x) \) mapping code substrings to \( y \in \{\text{BER3}, \text{POZ1}, \text{POZ2}, \ldots\} \) using frequency tables and direct string comparison.

2. **Layer 2: Merchant Lookup**
   - Uses merchant associations if tracking IDs relate to high-purity merchant→warehouse mappings (Bayesian confidence).
   - If \( P(\text{warehouse}|\text{merchant}) \gg 0.95 \), uses that with high confidence.

3. **Layer 3: Gradient Boosted Trees (GBT)**
   - For ambiguous IDs, applies a trained model (GBT/Decision Tree ensemble) to learn subtle pattern dependencies.
   - Features include tracking ID character n-grams, positional substrings, length, merchant identity, Markov chain transitions over strings, and more.
   - Each tree in the ensemble partitions input space \( X \) (feature vectors) via axis-aligned splits to maximize information gain relative to warehouse labels \( Y \).
   - Formal:
     \[
     \hat{y} = \sum_{t=1}^T \eta f_t(x)
     \]
     with \( T \) trees \( f_t \), learning rate \( \eta \).
   - The training algorithm:
     - Minimizes cross-entropy/log-loss over one-hot \( Y \)
     - Trained using batch gradient boosting on parsed CSVs of historical returns.

4. **Supplementary: Markov & N-gram Statistical Models**
   - Used to build string feature representations for tracking ID structure, improving predictive context for rare or novel IDs.

### Training Pipeline

- **CSV Upload**: User provides a CSV with columns: [TrackingID, Merchant, Warehouse]
- **Feature Extraction**: Script parses:
  - Prefix/suffix tables (frequency)
  - Character/word n-grams
  - Markov state transitions across codes
  - Merchant/warehouse conditional distributions
- **Model Fitting**:
  - Deterministic rule table generated for high-frequency patterns.
  - For ambiguous IDs, fits a GBT ensemble (typically with max depth 4–6; number of trees dynamically determined by data volume).
  - Parameters optimized using stratified k-fold for warehouse balance.
- **Model Export**: Trained model state is JSON-serialized; can be loaded for instant, reproducible predictions.

### Prediction Confidence Calculation

- Each prediction includes a confidence score \(\in [0,1]\) derived either from deterministic probability (rules) or tree ensemble probabilistic output (softmax across classes).

---

## Mathematical Model

#### Rule Engine:  
For input \( x \):
\[
y = 
\begin{cases}
    f_{\text{prefix}}(x), & \text{if prefix rule match} \\
    f_{\text{structural}}(x), & \text{if structural property matches (e.g., regex)} \\
    \text{undefined}, & \text{otherwise}
\end{cases}
\]

#### Tree Ensemble:
Given features \( \phi(x) \):
\[
P(y=k|x) = \text{softmax} \left( \sum_{t=1}^T f_t(\phi(x)) \right)
\]
where \( f_t \) is the \( t \)-th decision tree.

#### Confidence:
If multiple layers provide conflicting predictions, the system prioritizes higher-confidence sources, combining via:
\[
\text{Conf}(y) = \max (\text{RuleConf}, \text{ModelConf})
\]
falling back to model output if rules are ambiguous.

---

## Tech Stack

- **Frontend**: HTML5 / CSS / JavaScript (Vanilla, no frameworks)
- **Modeling**: Pure client-side JS for all machine learning logic and statistical processing.
- **No backend required:** All computation and data privacy is performed in-browser.

---

## Getting Started

### Quick Start

1. Clone this repository, or download the ZIP.
2. Open `index.html` in any modern browser.
3. Load your historic return data as a CSV via the sidebar.
4. The model will train and become ready for predictions—input or scan a tracking ID to route each parcel!

### Requirements

- Latest Chrome, Firefox, Edge, or Safari
- No additional dependencies

---

## Typical Use Case

1. **Upload Dataset**: Drag and drop a CSV of returns history.
2. **Train**: Wait for the sidebar progress bar.
3. **Predict**: Enter tracking ID → instant warehouse suggestion with confidence.
4. **Monitor Metrics**: View session stats, warehouse split, merchant breakdown, and recent predictions with confidence levels.

---

## License

This project is made available as open source for non-commercial, educational, and operational warehouse logistics enhancement.

---

## Author

**prabathSoft** ([GitHub Profile](https://github.com/prabathSoft))

---

_Last updated: May 2026_

For more, visit the [ReturnRack Repository](https://github.com/prabathSoft/ReturnRack)
