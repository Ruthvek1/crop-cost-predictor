# Project Summary – Simple Linear Regression on Fish Dataset

The aim of this project was to perform **Simple Linear Regression** on the Fish dataset to understand how different physical measurements relate to fish Weight and Length.

## Key Findings
- **Strong predictors of Weight:** Length1, Length2, and Length3 showed very high predictive power with R² > 0.83. Width also showed a strong relationship (R² = 0.7859).
- **High correlation among lengths:** Length1, Length2, and Length3 were almost perfectly correlated. Length1 vs Length2 gave an R² of 0.9990, indicating near-interchangeable measurements.
- **Height is a weaker predictor:** Height had lower R² values (0.5247 for Weight and 0.3911 for Length1).
- **Best model:** Length1 → Length2 regression performed the best with R² = 0.9990, low MSE (0.0958), MAE (0.2479), and RMSE (0.3096).

## What We Did
- Loaded and inspected the **Fish.csv** dataset.
- Selected numerical regression features: Weight, Length1, Length2, Length3, Height, Width.
- Performed **10 simple linear regressions**:
  - Weight vs each feature
  - Length1 vs each feature
- Created scatter plots with regression lines and R² values.
- Compared all R² scores using a bar chart.
- Conducted **correlation analysis** with a heatmap.
- Identified the best regression model and calculated detailed performance metrics.
- Performed model diagnostics with residual plots and distribution analysis.
