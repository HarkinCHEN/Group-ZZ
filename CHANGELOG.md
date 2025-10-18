# Changelog

## v0.2 - Experiment with RandomForest
- **Change**: Switched model from `LinearRegression` (v0.1) to `RandomForestRegressor` (v0.2).
- [cite_start]**Rationale**: Attempted to use a more complex ensemble model (per MLOps 2 slides) [cite: 254, 276, 278] with the expectation of capturing non-linear relationships and lowering the RMSE.
- **Metric Deltas**:
  - **v0.1 (LinearRegression) RMSE**: 53.85344583676594
v0.2 (RandomForest) RMSE: 
  - **v0.2 (RandomForest) RMSE**: 54.398350726816645
- **Conclusion**: On this dataset, the v0.2 (`RandomForestRegressor`) model's test RMSE was **higher** than the v0.1 (`LinearRegression`) baseline. This suggests the v0.1 baseline is very strong, or that the v0.2 model may be overfitting and requires further hyperparameter tuning.

## v0.1 - Baseline Model
- **Change**: Initial service implementation with a `StandardScaler` + `LinearRegression` pipeline.
- **Metric**:
  - **v0.1 (LinearRegression) RMSE**:  53.85344583676594
