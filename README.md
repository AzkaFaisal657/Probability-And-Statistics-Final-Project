# Probability and Statistics Final Project

An interactive **Dash** dashboard for exploring a student entrepreneurial projects dataset using probability, statistics, exploratory data analysis, and regression modeling.

## Overview

This project provides a web-based dashboard that helps users analyze how different factors relate to project competitiveness. It includes:

- Overview charts and summary metrics
- Frequency distributions for numeric and categorical variables
- Exploratory data analysis with box plots, histograms, confidence intervals, skewness, and coefficient of variation
- Probability distribution demos for Normal, Binomial, and Poisson distributions
- Regression analysis and prediction tools
- A built-in dataset viewer

## Dataset

The app uses:

- `student_entrepreneurial_projects.csv`

The dashboard analyzes variables such as:

- Competitiveness score
- Market size and growth rate
- Team size and experience
- Technical and business skill scores
- Mentor feedback score
- Social media mentions
- Education level
- Funding stage
- Project domain
- Innovation type

## Features

### 1. Overview Dashboard
- Filter data by domain, funding stage, and innovation type
- View project distribution, box plots, bar charts, and histograms
- See key summary cards for quick insights

### 2. Frequency & Distributions
- Build frequency tables and histograms for continuous variables
- Visualize categorical distributions with bar and pie charts
- Inspect cumulative and relative frequencies

### 3. EDA & Shape of Data
- Box plots with outlier detection
- Statistical summaries including mean, median, mode, variance, standard deviation, IQR, kurtosis, and skewness
- 95% confidence intervals for numeric variables
- Grouped box plots by education level
- Coefficient of variation comparisons

### 4. Probability Distributions
- Normal distribution calculator with probability shading and z-score
- Binomial distribution explorer
- Poisson distribution explorer

### 5. Regression & Predictions
- Simple linear regression explorer
- Multiple regression-based competitiveness prediction
- Interactive sliders for model inputs

### 6. Dataset Viewer
- View the raw dataset in a separate page route at `/view-data`

## Tech Stack

- Python
- Dash
- Plotly
- Pandas
- NumPy
- SciPy
- scikit-learn

## Installation

1. Clone the repository

```bash
git clone https://github.com/AzkaFaisal657/Probability-And-Statistics-Final-Project.git
cd Probability-And-Statistics-Final-Project
```

2. Create and activate a virtual environment

```bash
python -m venv venv
```

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

3. Install dependencies

```bash
pip install dash plotly pandas numpy scipy scikit-learn
```

## Run the App

```bash
python app.py
```

Then open the app in your browser at:

```text
http://127.0.0.1:8050
```

## Project Structure

```text
Probability-And-Statistics-Final-Project/
├── app.py
├── student_entrepreneurial_projects.csv
├── .gitignore
└── README.md
```

## Notes

- The dashboard is designed for interactive learning and project analysis in probability and statistics.
- The regression predictor uses a pre-trained linear regression model built from the dataset.
- The `/view-data` route displays the full dataset in a simple styled table.

## Author

Made for a Probability & Statistics final project.
