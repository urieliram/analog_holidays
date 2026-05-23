```python
# Implement the DynaNC preprocessing stage from the paper:
# “Holiday Load Forecasting Using DynaNC and LSTMH”
#
# OBJECTIVE:
# Build the DTW-based preprocessing pipeline used to generate
# holiday-aware inputs for the LSTM models.
#
# The goal is:
#
# 1. Identify the nearest workday before a holiday.
# 2. Find the most similar historical workday using DTW.
# 3. Generate adjusted load profiles using the DynaNC equations.
# 4. Construct training samples for the LSTM models.
#
# ==========================================================
# INPUT DATA
# ==========================================================
#
# Input dataframe columns:
#
# unique_id   : load zone identifier
# ds          : datetime timestamp hourly
# y           : electric load
# temp        : temperature
# holiday     : binary holiday indicator
# dow         : day of week (0=Mon,...6=Sun)
#
# Data frequency:
# hourly
#
# ==========================================================
# STEP 1 — BUILD DAILY MATRICES
# ==========================================================
#
# Convert hourly data into daily 24-hour vectors:
#
# Example:
#
# date         -> [24 hourly load values]
# load_vector  -> shape (24,)
# temp_vector  -> shape (24,)
#
# Build:
#
# daily_load[date]
# daily_temp[date]
#
# ==========================================================
# STEP 2 — IDENTIFY HOLIDAY TRANSITIONS
# ==========================================================
#
# For each holiday date H:
#
# define:
#
# nearest_workday W:
#     last non-holiday weekday before H
#
# Store:
#
# holiday_date
# nearest_workday
#
# ==========================================================
# STEP 3 — DTW SEARCH
# ==========================================================
#
# For each nearest workday W:
#
# Compare:
#
# W-7 days
# W-14 days
#
# using Dynamic Time Warping (DTW).
#
# Use only observed hours:
#
# Example:
#
# if current hour = delta
# use hours:
# [0 : delta-1]
#
# DTW recurrence:
#
# W[i,j] =
# abs(a[i]-b[j]) +
# min(
#     W[i-1,j],
#     W[i,j-1],
#     W[i-1,j-1]
# )
#
# Initialize:
#
# first row = inf
# first column = inf
# W[0,0] = 0
#
# Return:
#
# best_match_day
# dtw_distance
#
# ==========================================================
# STEP 4 — DynaNC LOAD ADJUSTMENT
# ==========================================================
#
# After selecting the most similar historical day:
#
# create adjusted forecast profile:
#
# forecast =
# analog_profile
# + offset
# + trend_adjustment
#
# Implement Equation (2):
#
# offset =
# sign(mean(current_observed - analog_observed))
# * O1
#
# trend adjustment:
#
# preserve analog slope continuity.
#
# Add configurable parameters:
#
# O1 = float
#
# ==========================================================
# STEP 5 — TEMPERATURE-BASED COMPENSATORY HOLIDAYS
# ==========================================================
#
# Implement Equations (5) and (6):
#
# Weekend compensatory holiday:
#
# forecast =
# similar_temperature_day
# + temperature_offset
#
# Weekday compensatory holiday:
#
# forecast =
# previous_day_load
# - DTW(temp_current,temp_previous)*O3
#
# Add configurable:
#
# O2
# O3
#
# ==========================================================
# STEP 6 — BUILD LSTM TRAINING FEATURES
# ==========================================================
#
# Construct supervised learning dataset:
#
# INPUT FEATURES:
#
# - previous 14 days hourly load
# - previous 14 days temperature
# - holiday indicator
# - day of week
# - hour of day
# - DTW distance
# - selected analog profile
# - adjusted DynaNC profile
#
# TARGET:
#
# 24-hour holiday load profile
#
# ==========================================================
# OUTPUTS
# ==========================================================
#
# Produce:
#
# 1. DTW matching function
# 2. Daily vector builder
# 3. Holiday transition detector
# 4. DynaNC adjustment functions
# 5. Feature engineering pipeline
# 6. Final pandas dataframe ready for NeuralForecast LSTM
#
# ==========================================================
# IMPLEMENTATION DETAILS
# ==========================================================
#
# Use:
#
# pandas
# numpy
# sklearn
# scipy
#
# Code should be:
#
# modular
# production-quality
# well-commented
#
# Avoid using external DTW libraries.
# Implement DTW manually from the recurrence equation.
#
# Include example usage at the end.
```
