# Data Science Research Capstone

## An End-to-End Reproducible Data Science Framework for Statistical Inference, Unsupervised Learning, and Time-Series Forecasting

### Abstract
This capstone synthesizes a reproducible workflow spanning statistical inference, PCA, unsupervised clustering, and ARIMA forecasting. The repository contains executed notebooks, source utilities, tests, reports, figures, and datasets. The supplied demand dataset contains 30 daily observations from 1 June through 30 June 2026. A 15-day training/15-day holdout design was therefore used, followed by a 30-day forecast beginning 1 July 2026.

### Research Question
How can a reproducible data science workflow combine statistical inference, unsupervised learning, and time-series forecasting to identify meaningful structure and support evidence-based forecasting decisions?

### Methods
Statistical analysis includes Shapiro–Wilk, Kolmogorov–Smirnov, Welch's t-test, Mann–Whitney U, one-way ANOVA, two-way ANOVA, and Tukey HSD. PCA is used for dimensionality reduction. K-Means, DBSCAN, and hierarchical clustering are compared using internal validation. ARIMA candidate models are evaluated using AIC and a holdout period.

### Forecasting Result
The first five forecast values are approximately 155.94, 158.02, 152.99, 149.84, and 151.57 for 1–5 July 2026.

### Research Integrity
The statistical and clustering datasets are synthetic/demo datasets. The demand series is the supplied forecasting dataset. Synthetic results are not presented as evidence about an external population.

### Limitations
The demand series is short, limiting long-horizon forecasting reliability. Internal clustering metrics do not establish business usefulness, and synthetic analyses do not establish external validity.

### Conclusion
The capstone demonstrates an end-to-end reproducible framework connecting research questions, statistical inference, unsupervised learning, forecasting, evaluation, and documentation.
