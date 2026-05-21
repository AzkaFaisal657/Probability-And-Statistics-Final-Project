# Student Entrepreneurial Projects Analysis Dashboard

An interactive, premium Plotly Dash analytical dashboard designed to explore student entrepreneurial competitiveness. This application utilizes rigorous probability distributions, descriptive statistics, and dynamic regression modeling to evaluate how team composition, market growth, and core skill sets influence project success.

---
## Key Dashboard Architecture

The dashboard is structured into five cohesive analytical layers:

### Tab 1: Overview Dashboard
* Provides high-level visual exploration of the dataset.
* Interactive filters for project domain, funding stage, and innovation type.
* Key performance metrics (total projects, average competitiveness, team sizes, and market growth rate).
* Distribution pie charts and competitiveness breakdowns.

### Tab 2: Frequency & Distributions
* **Quantitative Frequency Distribution:** Generates custom-binned frequency tables (class intervals, raw frequency, relative percentage, and cumulative frequency) alongside dynamic histograms.
* **Qualitative Frequency Distribution:** Explores categorical variables (funding stages, project domains, innovation types) with synchronous bar, pie, and tabular distributions.

### Tab 3: EDA & Shape of Data
* **Descriptive Stats Table:** Computes exact statistical summaries including Mean, Median, Mode, Variance, Standard Deviation, IQR, Kurtosis, and Skewness with dynamic skew classification.
* **Outlier Detection Box Plot:** Displays exact Q1, Q3, Median, and Whisker fences dynamically annotated inside the visualization.
* **Confidence Intervals (95%):** Displays a structured statistical table showing the 95% Confidence Interval width, lower bounds, and upper bounds for every numeric variable in the dataset.
* **Correlation Matrix (Section B3):** A custom-designed heat map visualizing Pearson correlation coefficients between all numeric variables, mapped to a tailored diverging color scale, paired with a ranked horizontal correlation bar chart.

### Tab 4: Probability Distributions
* **Normal Distribution:** Interactive simulation of Z-scores, PDF curves, and shaded CDF regions based on configurable mean ($\mu$) and standard deviation ($\sigma$).
* **Binomial Distribution:** Simulates discrete probability mass functions (PMF) and CDFs based on the number of trials ($n$) and success probability ($p$).
* **Poisson Distribution:** Models random independent event intervals using a configurable average occurrence rate ($\lambda$).

### Tab 5: Regression & Predictions
* **Simple Linear Regression:** Evaluates single-variable relationships with line-of-best-fit plotting, R-squared values, p-values, and a dynamic color-coded **Correlation Strength** badge.
* **Multiple Regression Predictor:** Trains a multi-variable Scikit-Learn `LinearRegression` model.
  * **Dynamic Predictors:** Automatically evaluates all numeric variables and keeps only features with a Pearson correlation coefficient $|r| \ge 0.1$ with competitiveness.
  * **Interactive Scenarios:** Generates a real-time prediction using only the valid features, dynamically showing/hiding sliders accordingly.

---

## Mathematical and Statistical Foundations

The calculations driving the dashboard rely on the following mathematical formulations:

### Pearson Correlation Coefficient ($r$)
Used to identify linear relationships and dynamically select predictors:
$$r = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum (X_i - \bar{X})^2 \sum (Y_i - \bar{Y})^2}}$$

### Simple Linear Regression
Model used in Section A:
$$\hat{Y} = \beta_0 + \beta_1 X$$

### Multiple Linear Regression
Model used in Section B (where $X_j$ are features with $|r| \ge 0.1$):
$$\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_k X_k$$

---

## Setup and Installation

Follow these steps to run the application locally on your machine.

### Prerequisites
* Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository
Open your terminal and navigate to the project directory:
```bash
cd PROBABILITY-AND-STATISTICS
```

### Step 2: Set Up the Virtual Environment
Create and activate an isolated python virtual environment to avoid dependency conflicts:

**For Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**For Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required libraries inside the virtual environment:
```bash
pip install -r requirements.txt
```
*(If you do not have a requirements file, the core packages required are `dash`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `plotly`, and `gunicorn`)*

### Step 4: Run the Application
Launch the Flask development server:

**Using Activated Environment:**
```bash
python app.py
```

**Direct Method (No Activation Required):**
```bash
venv\Scripts\python app.py
```

The terminal will confirm that the application is running. Open your browser and navigate to:
```text
http://localhost:8050/
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