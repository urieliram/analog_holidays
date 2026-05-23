# Implement the LSTMH methodology from the paper:
# “Holiday Load Forecasting Using DynaNC and LSTMH”
# using Nixtla NeuralForecast library with PyTorch backend.
#
# OBJECTIVE:
# Build two LSTM forecasting models:
#
# 1. U-LSTM (Univariate LSTM)
#    - Single model predicts the full 24-hour holiday profile.
#    - Learns interdependency between hours.
#
# 2. M-LSTM (Multivariate / per-hour LSTM)
#    - Train 24 independent models.
#    - Each model predicts one specific hour.
#
# DATA:
# Input dataframe columns:
#
# unique_id   : region/load zone identifier
# ds          : datetime hourly timestamp
# y           : load
# temp        : temperature
# holiday     : binary holiday indicator
# dow         : day of week (0-6)
# hour        : hour of day (0-23)
#
# REQUIREMENTS:
#
# ---------------------------------------
# PREPROCESSING
# ---------------------------------------
#
# 1. Use MinMaxScaler for:
#    - y
#    - temp
#
# 2. One-hot encode:
#    - dow
#    - holiday
#    - hour
#
# 3. Build rolling windows using:
#    - previous 14 days (336 hours)
#
# ---------------------------------------
# U-LSTM MODEL
# ---------------------------------------
#
# Build a single LSTM model that:
#
# INPUTS:
# - previous 336 hours
# - features:
#     y
#     temp
#     dow
#     holiday
#     hour
#
# OUTPUT:
# - next 24 hourly load values
#
# Suggested architecture:
#
# LSTM(
#     hidden_size=64,
#     num_layers=2,
#     dropout=0.2
# )
#
# followed by:
# Dense(32)
# Dense(24)
#
# Use:
# - MAE loss
# - Adam optimizer
# - early stopping
#
# Use NeuralForecast library.
#
# ---------------------------------------
# M-LSTM MODEL
# ---------------------------------------
#
# Create 24 independent LSTM models:
#
# One model per forecast hour.
#
# Example:
# - model_h00 predicts hour 0
# - model_h01 predicts hour 1
# ...
#
# Each model uses:
# - same historical features
# - only predicts one scalar output
#
# Train all models in a loop.
#
# ---------------------------------------
# ENSEMBLE
# ---------------------------------------
#
# After obtaining:
#
# yhat_u  -> prediction from U-LSTM
# yhat_m  -> prediction from M-LSTM
#
# Train a simple ensemble meta-model:
#
# Input:
# [yhat_u, yhat_m]
#
# Output:
# true load
#
# Use:
# - LinearRegression
# OR
# - small MLPRegressor
#
# ---------------------------------------
# EVALUATION
# ---------------------------------------
#
# Evaluate:
#
# - MAE
# - RMSE
# - sMAPE
#
# specifically on holidays only.
#
# ---------------------------------------
# HOLIDAY FILTERING
# ---------------------------------------
#
# During training:
# only use windows where:
#
# - target day is holiday
# OR
# - previous day before holiday
#
# ---------------------------------------
# OUTPUTS
# ---------------------------------------
#
# Produce:
#
# 1. Training pipeline
# 2. Feature engineering pipeline
# 3. NeuralForecast implementation
# 4. Prediction pipeline
# 5. Evaluation metrics
# 6. Example forecast plot for one holiday
#
# Use clean modular Python code.
# Use pandas, sklearn, matplotlib and neuralforecast.
#
# IMPORTANT:
# The implementation should work for hourly electric load forecasting.