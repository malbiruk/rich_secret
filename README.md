# Rich Secret 💰

**Rich Secret** is a personal budgeting web application built using Streamlit. It helps you track, plan, and analyze your finances with ease, leveraging a Google Sheets template.

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live_App-brightgreen)](https://rich-se-cret.streamlit.app/)
[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Features

### Setup
1. **Google Sheets Integration**:
   - Make a copy of the provided [Google Sheets Template](https://docs.google.com/spreadsheets/d/19wTUH2nv4bkI2fPUgGaLDiFV9tyY3N1j3hBoVx2zBsk) and populate it with your data.
   - Ensure the sheet is shareable (`Anyone with the link can view`).
   - Input the link to your sheet into the app.

2. **Template Sheets Overview**:
   - **`monthly_plan`**: Plan your monthly expenses, income, and savings.
   - **`expenses`, `income`, `savings`**: Track actual financial data.
   - **`categories`**: Data validation and customization.
   - **`init`**: Initial balance and savings data.

   *(Detailed instructions can be found in the README within the Google Sheets template.)*

### App Functionalities
1. **Settings Section**:
   - Select a time mode (Month, Quarter, Year, or Custom to specify date range).
   - Choose a display currency.
   - Aggregate data by 1, 3, 7, 14, or 30 days in visualizations.
   - Option to hide Fixed expenses in visualizations.

2. **Stats Section**:
     - Balance, total income, total expenses, total savings.
     - Weekly spend/allowance.
     - Changes from the previous period in both absolute and percentage terms.

3. **Plots (Powered by Plotly)**:
   - Dumbbell plots comparing planned vs actual amounts for expenses, income, and savings.
   - Aggregated time-series plots:
     - **Ridgeline-like Plot**: Expenses distribution by category.
     - **Balance Overview**: Lineplot of balance with lollipops showing changes to balance by income, expenses, and savings.
     - **Savings Overview**: Stacked area plot of savings by category.

  All plots are interactive and feature detailed hover tooltips.

---

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rich-secret.git
   cd rich-secret
   ```
2. Install dependencies (Python 3.12):
  ```bash
  pip install -r requirements.txt
  ```
3. Configure the currency conversion API:
  - Register at [fxratesapi.com](https://fxratesapi.com/) to obtain an API key.
  - Save the key in `.streamlit/secrets.toml` located in the same folder as `app.py`:
    ```bash
    fxrates_api = "fxr_live_{your_api_key}"
    ```
4. Run the app:
  ```bash
  streamlit run app.py
  ```
5. Access the app at `http://localhost:8501`.

P.S. For local use you can set up a Google Sheets link in `.env`:
```
SHEETS_LINK = https://docs.google.com/spreadsheets/d/your_sheets_id/edit?usp=sharing
```
