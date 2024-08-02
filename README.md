# DeepScatter: A Intelligent Approach to Unveiling Drift Anomalies

* **Authors**: Luca Podo, Alessandro Di Patria
* **Emails**: podo@di.uniroma1.it, dipatria.1844538@studenti.uniroma1.it
* **Institution**: Sapienza University of Rome

## Table of Contents

1. [Introduction](#introduction)
2. [Background](#background)
   1. [Drift anomaly](#drift)
   2. [What is an autoencder?](#autoencoder)
3. [Methodology](#methodology)
   1. [Description of the input dataset](#read)
   2. [Data transformation](#tranformation)
   3. [Autoencoder architecture](#architecture)
   4. [Profile similarity comparison](#comparison)
4. [Conclusion and result](#conclusion_and_results)
5. [References](#references)


## Introduction  <a name="introduction"></a>
Discovering and understanding shift anomalies in time series data can be challenging due to their persistent deviations from the usual trend [\[3\]](#references). Unlike isolated point anomalies, shift anomalies disrupt the overall pattern, making their identification crucial across different domains. In this blog post, we present a novel visual approach that combines a simple deep learning model, specifically an autoencoder, with scatter plots to enhance the visualization and detection of shift anomalies.


Here, we leverage the power of deep learning models to capture complex patterns and representations in data. By integrating deep learning algorithms, visual analytics techniques, and domain expert decision processes, we provide a comprehensive solution [\[2\]](#references). The compact architecture of the autoencoder effectively learns the latent features of the time series, enabling a better understanding and identification of shift anomalies [\[1\]](#references). Moreover, the autoencoder's ability to reconstruct the input data brings the discrepancies caused by these anomalies into focus, making them visually distinguishable in scatter plots.

We propose to take advantage of human observers' inherent visual perception capabilities, enhancing the interpretability of shift anomaly detection by emphasizing anomalies and reducing the emphasis on normality. Our approach facilitates more accurate and context-aware decision-making by empowering domain experts to visualize and analyze anomalies visually. **It serves as a supportive visual tool for exploratory data analysis rather than an automated anomaly detection model.**

## Background <a name="background"></a>

Drift anomalies in time series data refer to significant changes or shifts in the underlying pattern or behavior of a sequence of data points over time. These anomalies can occur gradually or abruptly, posing challenges for accurate prediction, analysis, and decision-making processes.

**Shift Anomalies Across Domains**:
Drift anomalies can manifest in various domains, including finance, sensor data analysis, network monitoring, and industrial processes. They can arise from unexpected events, system failures, human errors, or external factors that introduce abrupt changes to the underlying dynamics of the time series data.

**Real-World Shift Anomaly: The Heatwave Scenario**:
To better comprehend drift anomalies, let's consider an example regarding measurement of temperature values in a region known for its stable climate and consistent seasonal temperature variations (e.g., the Amazon rainforest). Suddenly, a high-pressure system settles over the area, causing a heatwave. As a result, the region experiences an unexpected and drastic increase in temperature for an extended period, surpassing historical records for that particular season. *This shift anomaly leads to extreme heat, triggering challenges such as heatwaves, increased energy demands, and potential risks to human health and the environment.* This example highlights the broader implications of drift anomalies beyond financial contexts.

![shift 1.png](data:image/png;base64,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)
