# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an INFS7203 Data Mining coursework repository containing machine learning datasets for classification tasks.

## Data Structure

- `data/train.csv` - Training dataset (~10,854 rows) with 43 features and binary target variable
- `data/test_data.csv` - Test dataset (~2,714 rows) with 43 features (no target column)

### Dataset Schema

- **Numerical features**: Num_Col1 through Num_Col25 (25 columns)
- **Categorical features**: Nom_Col26 through Nom_Col43 (18 columns)
- **Target variable**: Target (Col44) - binary classification (0/1)
- **Missing values**: Present throughout the dataset (represented as NaN)

## Development Commands

This repository contains only CSV data files. Common operations would typically involve:

- **Data exploration**: Use pandas to load and examine the datasets
- **Model development**: Standard Python ML libraries (scikit-learn, pandas, numpy)
- **Jupyter notebooks**: Likely development environment for data analysis

No specific build, test, or lint commands are configured as this appears to be a data-only repository without Python package structure.

## Architecture Notes

- Simple data repository structure with train/test split
- Mixed data types (numerical and categorical) requiring preprocessing
- Binary classification problem suitable for various ML algorithms
- Missing data handling will be required for most analyses
- 使用jupyter notebook