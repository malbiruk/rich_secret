"""
Rich Secret App
budget overview.
"""

import datetime
import re
import urllib

import numpy as np
import one_light_template  # noqa: F401
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import requests
import streamlit as st
from millify import millify
from plotly.subplots import make_subplots
from ridgeline_plot import color_to_rgb, ridgeline

pio.templates.default = "one_light"

AGGREGATE_BY_TO_FREQ = {
    "days": "1D",
    "weeks": "1W",
    "months": "30D",
}


@st.cache_data
def get_data(google_doc_id: str) -> dict[str, pd.DataFrame]:
    budget_sheets = pd.read_excel(
        f"https://docs.google.com/spreadsheets/d/{google_doc_id}/export?format=xlsx",
        sheet_name=None)

    categorical_columns = {
        "monthly_plan": ["type", "category", "currency"],
        "expenses": ["category", "currency"],
        "income": ["category", "currency"],
        "savings": ["category", "currency"],
        "init": ["currency"],
    }

    datetime_columns = (
        "monthly_plan",
        "expenses",
        "income",
        "savings",
    )

    budget_sheets["monthly_plan"]["type"] = budget_sheets["monthly_plan"]["type"].str.lower()

    for sheet in datetime_columns:
        budget_sheets[sheet]["date"] = pd.to_datetime(budget_sheets[sheet]["date"])

    for sheet, columns in categorical_columns.items():
        for column in columns:
            budget_sheets[sheet][column] = pd.Categorical(budget_sheets[sheet][column])

    return budget_sheets


def get_conversion_rate(row, target_currency: str, exchange_rates: dict[str, float]) -> float:
    current_currency = row["currency"]
    if current_currency in exchange_rates and target_currency in exchange_rates:
        exchange_rate = exchange_rates[target_currency] / exchange_rates[current_currency]
        return row["amount"] * exchange_rate
    return None


def get_previous_period(start_date, end_date, aggregate_by):
    if aggregate_by == "days":
        prev_start_date = (start_date - pd.DateOffset(months=1)).replace(day=1)
        prev_end_date = (prev_start_date + pd.DateOffset(months=1)) - pd.Timedelta(days=1)
    elif aggregate_by == "weeks":
        prev_start_date = (start_date - pd.DateOffset(months=3)).replace(day=1)
        prev_end_date = (prev_start_date + pd.DateOffset(months=3)) - pd.Timedelta(days=1)
    elif aggregate_by == "months":
        prev_start_date = start_date - pd.DateOffset(years=1)
        prev_end_date = end_date - pd.DateOffset(years=1)
    else:
        raise ValueError("Invalid agrregate_by type. Use 'days', 'weeks', or 'months'.")

    return prev_start_date, prev_end_date


@st.cache_data
def convert_amounts_to_target_currency(
        budget_sheets: dict[str, pd.DataFrame],
        target_currency: str,
        exchange_rates: dict[str, float]) -> dict[str, pd.DataFrame]:
    for df in budget_sheets.values():
        if "amount" in df.columns and "currency" in df.columns:
            df["converted_amount"] = df.apply(
                get_conversion_rate,
                axis=1,
                target_currency=target_currency,
                exchange_rates=exchange_rates)
    return budget_sheets


@st.cache_data
def filter_sheets_by_date_range(
        budget_sheets: dict[str, pd.DataFrame],
        start_date: datetime.date,
        end_date: datetime.date) -> dict[str, pd.DataFrame]:
    for sheet in ["monthly_plan", "expenses", "income", "savings"]:
        budget_sheets[sheet] = budget_sheets[sheet][
            (budget_sheets[sheet]["date"] >= start_date)
            & (budget_sheets[sheet]["date"] <= end_date)]
    return budget_sheets


@st.cache_data
def get_exchange_rates():
    response = requests.get(
        f"https://api.fxratesapi.com/latest?api_key={st.secrets['fxrates_api']}")
    exchange_rates = response.json()["rates"]
    exchange_rates["USDT"] = 1
    return exchange_rates


def initialize_session_state():
    if "google_doc_id" not in st.session_state:
        st.session_state.google_doc_id = None
    if "first_run" not in st.session_state:
        st.session_state.first_run = True
    if "budget" not in st.session_state:
        st.session_state.budget = None
    if "modified_budget" not in st.session_state:
        st.session_state.modified_budget = None
    if "target_currency" not in st.session_state:
        st.session_state.target_currency = None
    if "exchange_rates" not in st.session_state:
        st.session_state.exchange_rates = None


def customize_page_appearance() -> None:
    st.set_page_config(
        page_title="Rich Secret",
        page_icon="💰")

    st.markdown("""
        <style>
            [data-testid="stDecoration"] {
                display: none;
            }
            [data-testid="stMainBlockContainer"] {
            max-width: 1200px;
            padding-left: 50px;
            padding-right: 50px;
            }
        </style>
        """, unsafe_allow_html=True)


def settings(budget_sheets, now_gmt4):
    col1, col2, _col4, col3 = st.columns([1, 1, .5, .5])
    mode = col1.pills(
        "Mode",
        ["Month", "Quarter", "Year"],
        selection_mode="single",
        default="Month",
    )
    if mode is None:
        mode = "Month"
        col1.caption('Showing "Month" mode.')

    with col2:
        if mode == "Month":
            col_1, col_2 = st.columns(2)
            selected_month = col_1.selectbox(
                "Select Month", [f"{i:02d}" for i in range(1, 13)], index=now_gmt4.month - 1)
            selected_year = col_2.number_input(
                "Select Year", value=now_gmt4.year, min_value=2024, max_value=2100)
            start_date = datetime.date(int(selected_year), int(selected_month), 1)
            end_date = (start_date + datetime.timedelta(days=31)
                        ).replace(day=1) - datetime.timedelta(days=1)
            aggregate_by = "days"
        elif mode == "Quarter":
            col_1, col_2 = st.columns(2)
            selected_quarter = col_1.selectbox(
                "Select Quarter", ["Q1", "Q2", "Q3", "Q4"], index=(now_gmt4.month-1)//3)
            selected_year = col_2.number_input(
                "Select Year", value=now_gmt4.year, min_value=2024, max_value=2100)
            quarter_start_month = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}[selected_quarter]
            start_date = datetime.date(selected_year, quarter_start_month, 1)
            end_date = (start_date + datetime.timedelta(days=92)
                        ).replace(day=1) - datetime.timedelta(days=1)
            aggregate_by = "weeks"
        elif mode == "Year":
            _col, col_1, _col = st.columns([0.5, 1, 0.5])
            selected_year = col_1.number_input(
                "Select Year", value=now_gmt4.year, min_value=2024, max_value=2100)
            start_date = datetime.date(selected_year, 1, 1)
            end_date = datetime.date(selected_year, 12, 31)
            aggregate_by = "months"

    start_date = pd.to_datetime(start_date).floor("D")
    end_date = pd.to_datetime(end_date).floor("D")
    col2.caption(f"<p style='text-align: center;'>Date range: {start_date.strftime('%d.%m.%Y')}"
                 f" to {end_date.strftime('%d.%m.%Y')}</p>", unsafe_allow_html=True)

    currencies = (budget_sheets["categories"]
                  [budget_sheets["categories"]["type"] == "Currency"]
                  ["category"].unique())
    target_currency = col3.selectbox("Currency:", currencies, index=0)

    return target_currency, start_date, end_date, aggregate_by


def calculate_stats(budget_data, start_date, end_date):
    monthly_plan_data = budget_data["monthly_plan"]

    total_income = budget_data["income"]["converted_amount"].sum()
    total_expenses = budget_data["expenses"]["converted_amount"].sum()
    total_savings = (budget_data["savings"]["converted_amount"].sum()
                     + budget_data["init"].loc[1:, "converted_amount"].sum())

    balance = (
        budget_data["init"].loc[0, "converted_amount"]
        + total_income
        - total_expenses
        - total_savings + budget_data["init"].loc[1:, "converted_amount"].sum()
    )

    planned_income = (
        monthly_plan_data[monthly_plan_data["type"] == "income"]["converted_amount"].sum())
    fixed_expenses = (
        monthly_plan_data
        [(monthly_plan_data["type"] == "expense")
            & (monthly_plan_data["category"] == "Fixed")]["converted_amount"].sum())
    planned_savings = (
        monthly_plan_data[monthly_plan_data["type"] == "savings"]["converted_amount"].sum())

    n_weeks = (
        (budget_data["expenses"]["date"].max()
            - start_date
         ).days + 1) / 7
    fixed_actual_expenses = (
        budget_data["expenses"][budget_data["expenses"]["category"] == "Fixed"]
        ["converted_amount"].sum()
    )
    actual_weekly_spend = (total_expenses-fixed_actual_expenses) / n_weeks

    n_weeks = ((end_date - start_date).days + 1) / 7
    can_spend_weekly = (planned_income - fixed_expenses - planned_savings) / n_weeks
    return {
        "balance": balance,
        "total_income": total_income,
        "total_expenses": total_expenses,
        "total_savings": total_savings,
        "actual_weekly_spend": actual_weekly_spend,
        "can_spend_weekly": can_spend_weekly,
    }


def stats(aggregate_by, start_date, end_date):
    last_col_width = 1.5 if st.session_state.target_currency == "BTC" else 1.1
    cols = st.columns([1, 1, 1, 1, last_col_width])
    this_period_stats = calculate_stats(
        st.session_state.modified_budget, start_date, end_date)

    prev_period = get_previous_period(start_date, end_date, aggregate_by)
    prev_modified_budget = filter_sheets_by_date_range(st.session_state.budget, *prev_period)
    prev_modified_budget = convert_amounts_to_target_currency(
        prev_modified_budget, st.session_state.target_currency, st.session_state.exchange_rates)
    prev_period_stats = calculate_stats(prev_modified_budget, *prev_period)

    deltas = {
        "balance": this_period_stats["balance"] - prev_period_stats["balance"],
        "total_income": this_period_stats["total_income"] - prev_period_stats["total_income"],
        "total_expenses": this_period_stats["total_expenses"] - prev_period_stats["total_expenses"],
        "total_savings": this_period_stats["total_savings"] - prev_period_stats["total_savings"],
        "actual_weekly_spend": this_period_stats["actual_weekly_spend"] - prev_period_stats["actual_weekly_spend"],
        "can_spend_weekly": this_period_stats["can_spend_weekly"] - prev_period_stats["can_spend_weekly"],
    }

    precision = (5 if st.session_state.target_currency == "BTC"
                 else 0 if st.session_state.target_currency == "RUB"
                 else 2)
    metric_names = ["Balance", "Total Income", "Total Expenses",
                    "Total Savings", "Weekly Spend / Allowance"]
    for col, metric_name, this_period_value, delta in zip(
            cols[:4],
            metric_names[:4],
            list(this_period_stats.values())[:4],
            list(deltas.values())[:4],
            strict=True):
        col.metric(metric_name,
                   millify(this_period_value, precision=precision),
                   delta=millify(delta, precision=precision),
                   delta_color="inverse" if metric_name == "Total Expenses" else "normal",
                   help=(str(round(this_period_value, 10))
                         if st.session_state.target_currency == "BTC"
                         else str(round(this_period_value, 2))))

    actual_weekly_spend_full = (round(this_period_stats["actual_weekly_spend"], 10)
                                if st.session_state.target_currency == "BTC"
                                else str(round(this_period_stats["actual_weekly_spend"], 2)))
    expected_weekly_spend_full = (round(this_period_stats["can_spend_weekly"], 10)
                                  if st.session_state.target_currency == "BTC"
                                  else str(round(this_period_stats["can_spend_weekly"], 2)))

    this_period_stats["actual_weekly_spend"] = 0 if np.isnan(
        this_period_stats["actual_weekly_spend"]) else this_period_stats["actual_weekly_spend"]
    deltas["actual_weekly_spend"] = this_period_stats["actual_weekly_spend"] if np.isnan(
        deltas["actual_weekly_spend"]) else deltas["actual_weekly_spend"]

    cols[-1].metric(
        metric_names[-1],
        f"{millify(this_period_stats["actual_weekly_spend"], precision=precision)} / "
        f"{millify(this_period_stats["can_spend_weekly"], precision=precision)}",
        delta=(
            f"{millify(deltas["actual_weekly_spend"], precision=precision)} / "
            f"{millify(deltas["can_spend_weekly"], precision=precision)}"
        ),
        delta_color="off",
        help=f"Actual Weekly Spend (without fixed): {actual_weekly_spend_full}\n\n"
        f"Expected Weekly Spend (without fixed): {expected_weekly_spend_full}")


def create_plot_data_actual_vs_planned(
        plan_data,
        data_type,
        *,
        show_fixed: bool = False):

    plan_data = plan_data[plan_data["type"] == data_type].copy()
    actual_data = st.session_state.modified_budget[data_type].copy()
    if data_type == "expenses" and not show_fixed:
        if "Fixed" in actual_data["category"].cat.categories:
            actual_data["category"] = actual_data["category"].cat.remove_categories("Fixed")
        if "Fixed" in plan_data["category"].cat.categories:
            plan_data["category"] = plan_data["category"].cat.remove_categories("Fixed")
    planned_sums = (
        plan_data[plan_data["type"] == data_type]
        .groupby("category", observed=True)["converted_amount"]
        .sum()
        .reset_index(name="planned_amount"))
    actual_sums = (
        actual_data
        .groupby("category", observed=True)["converted_amount"]
        .sum()
        .reset_index(name="actual_amount"))
    plot_data = planned_sums.merge(actual_sums, on="category", how="outer")
    plot_data["actual_amount"] = plot_data["actual_amount"].fillna(0)
    plot_data["planned_amount"] = plot_data["planned_amount"].fillna(0)
    return plot_data.sort_values("actual_amount")


def add_actual_vs_planned_subplot(fig, plot_data, row, col, *, swapped_colors=False,
                                  opacity=1, marker_size=10, line_width=.5):
    # draw lines
    for _ind, df_row in plot_data.iterrows():
        if swapped_colors:
            line_color = ("#09ab3b"
                          if df_row["actual_amount"] > df_row["planned_amount"]
                          else "#ff2b2b")
        else:
            line_color = ("#ff2b2b"
                          if df_row["actual_amount"] > df_row["planned_amount"]
                          else "#09ab3b")
        fig.add_trace(
            go.Scatter(
                x=[df_row["planned_amount"], df_row["actual_amount"]],
                y=[df_row["category"], df_row["category"]],
                mode="lines",
                opacity=opacity,
                line=dict(color=line_color, width=line_width),
                showlegend=False),
            row=row, col=col)
        # show planned and actual difference on hover
        mid_x = (df_row["planned_amount"] + df_row["actual_amount"]) / 2
        mid_y = df_row["category"]
        fig.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="markers",
                hovertemplate="%{customdata}<extra>Planned - Actual</extra>",
                customdata=[df_row["planned_amount"] - df_row["actual_amount"]],
                opacity=0,
                marker=dict(color=line_color),
                showlegend=False,
            ),
            row=row, col=col)

    # draw planned scatter
    fig.add_trace(
        go.Scatter(
            x=plot_data["planned_amount"],
            y=plot_data["category"],
            mode="markers",
            name="Planned",
            showlegend=False,
            opacity=opacity,
            marker=dict(color="darkgray", size=marker_size)),
        row=row, col=col)

    # draw actual scatter
    fig.add_trace(
        go.Scatter(
            x=plot_data["actual_amount"],
            y=plot_data["category"],
            mode="markers",
            name="Actual",
            opacity=opacity,
            showlegend=False,
            marker=dict(color="lightseagreen", size=marker_size)),
        row=row, col=col)

    return fig


def actual_vs_planned_plot(show_fixed):
    st.markdown("<h5><span style='color:lightseagreen'>Actual</span>"
                " vs <span style='color:darkgray'>Planned</span></h5>",
                unsafe_allow_html=True)

    plot_data = {
        data_type: create_plot_data_actual_vs_planned(
            st.session_state.modified_budget["monthly_plan"],
            data_type,
            show_fixed=show_fixed,
        ) for data_type in ("expenses", "income", "savings")
    }

    fig = make_subplots(rows=2, cols=2,
                        specs=[[{"rowspan": 2}, {}],
                               [None, {}]],
                        vertical_spacing=0.25,
                        subplot_titles=("Expenses", "Income", "Savings"))

    fig = add_actual_vs_planned_subplot(fig, plot_data["expenses"], row=1, col=1)
    fig = add_actual_vs_planned_subplot(fig, plot_data["income"], swapped_colors=True, row=1, col=2)
    fig = add_actual_vs_planned_subplot(
        fig, plot_data["savings"], swapped_colors=True, row=2, col=2)

    fig.update_layout(template="one_light",
                      plot_bgcolor="#fafafa",
                      dragmode="pan",
                      margin={"l": 100, "r": 0, "t": 30, "b": 50},
                      )
    fig.update_annotations(font=dict(size=18), yshift=10)
    fig.update_xaxes(title=st.session_state.target_currency,
                     showline=False, showgrid=True, tickfont=dict(size=14))
    fig.update_xaxes(title=None, row=1, col=2)
    fig.update_yaxes(showline=False, tickfont=dict(size=14))

    st.plotly_chart(fig, theme=None)


def add_expenses_ridgeline(show_fixed, aggregate_by, start_date, end_date, fig, row, col):
    expenses_data = st.session_state.modified_budget["expenses"].copy()
    if not show_fixed and "Fixed" in expenses_data["category"].cat.categories:
        expenses_data["category"] = expenses_data["category"].cat.remove_categories("Fixed")
    category_sums = (expenses_data.groupby("category", observed=True)["converted_amount"].sum())
    sorted_categories = category_sums.sort_values().index

    ridgeline(expenses_data,
              category_col="category",
              categories_order=sorted_categories,
              data_col="date",
              stats_col="converted_amount",
              normalize_each=False,
              scaling_factor=1.5,
              edgecolor="dimgray",
              hover_stats=False,
              points_size=5,
              jitter_strength_x=1,
              jitter_strength_y=1,
              line_width=1.5,
              hoverdata=["name", "amount", "currency"],
              bin_width=AGGREGATE_BY_TO_FREQ[aggregate_by],
              data_range=(start_date, end_date),
              ylabels_xref="paper",
              fig=fig, row=row, col=col)

    return fig


def add_balance_lineplot(aggregate_by, fig, row, col):
    freq = AGGREGATE_BY_TO_FREQ[aggregate_by]

    expenses = st.session_state.modified_budget["expenses"].copy()
    income = st.session_state.modified_budget["income"].copy()
    savings = st.session_state.modified_budget["savings"].copy()

    expenses["type"] = "Expense"
    income["type"] = "Income"
    savings["type"] = "Savings"

    expenses["converted_amount"] *= -1  # Expenses reduce balance
    savings["converted_amount"] *= -1  # Savings reduce balance

    all_data = pd.concat([expenses, income, savings])

    # aggregate data by day and type
    all_data = all_data.groupby([pd.Grouper(key="date", freq=freq), "type"]).agg(
        {"converted_amount": "sum"}).reset_index()

    # sum transactions per day and compute cumulative balance
    initial_balance = st.session_state.modified_budget["init"].loc[0, "converted_amount"]
    daily_totals = all_data.groupby("date")["converted_amount"].sum().reset_index()
    daily_totals["balance"] = initial_balance + daily_totals["converted_amount"].cumsum()

    # merge daily balances back into the transaction data
    all_data = all_data.merge(daily_totals[["date", "balance"]], on="date", how="left")

    # create balance data for plotting (calculate balance for each day)
    if pd.isna(all_data["date"].min()):
        st.rerun()
    date_range = pd.date_range(start=all_data["date"].min()-pd.to_timedelta(freq),
                               end=all_data["date"].max(), freq=freq)
    balance_data = daily_totals.set_index("date").reindex(date_range)
    balance_data["balance"] = balance_data["balance"].ffill()  # Fill missing days
    balance_data = balance_data.reset_index(names="date")
    balance_data["date"] = balance_data["date"] + pd.to_timedelta(freq)
    balance_data["converted_amount"] = balance_data["converted_amount"].fillna(0)
    balance_data.loc[0, "balance"] = (balance_data.loc[1, "balance"]
                                      - balance_data.loc[1, "converted_amount"])

    # calculate shifted balances for plotting lollipops
    shifted_data = all_data.copy()
    daily_balance = all_data.groupby("date")["balance"].first()
    daily_balance_shifted = daily_balance.shift(1)
    shifted_data["balance"] = shifted_data["date"].map(daily_balance_shifted)
    shifted_data.loc[0, "balance"] = balance_data.loc[0, "balance"]
    shifted_data["balance"] = shifted_data["balance"].ffill()

    for data_type, color in zip(["Expense", "Income", "Savings"][::-1],
                                ["lightsalmon", "steelblue", "plum"][::-1]):
        type_data = shifted_data[shifted_data["type"] == data_type]

        # Add "sticks" (lines from balance to value points)
        for _index, row_data in type_data.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row_data["date"], row_data["date"]],
                    y=[row_data["balance"], row_data["balance"] + row_data["converted_amount"]],
                    mode="lines",
                    legendgroup=data_type,
                    line=dict(color=color, width=.5),
                    showlegend=False),
                row=row, col=col)

        # Add "heads" (lollipop points)
        fig.add_trace(
            go.Scatter(
                x=type_data["date"],
                y=type_data["balance"] + type_data["converted_amount"],
                mode="markers",
                marker=dict(color=color, size=6, symbol="circle"),
                legendgroup=data_type,
                hovertemplate="Date: %{x}<br>Amount: %{customdata} {{v}}".replace(
                    "{{v}}", st.session_state.target_currency),
                customdata=(
                    type_data["converted_amount"] if st.session_state.target_currency == "BTC"
                    else round(type_data["converted_amount"], 2)),
                name=data_type),
            row=row, col=col)

    fig.add_trace(
        go.Scatter(
            x=balance_data["date"],
            y=(balance_data["balance"] if st.session_state.target_currency == "BTC"
               else round(balance_data["balance"], 2)),
            hovertemplate="Date: %{x}<br>Balance: %{y} {{v}}".replace(
                "{{v}}", st.session_state.target_currency),
            mode="lines",
            name="Balance",
            showlegend=False,
            line=dict(color="lightseagreen", width=2),
            line_shape="spline",
            line_smoothing=.9,
        ),
        row=row, col=col)

    return fig, all_data["date"].min(), all_data["date"].max()+pd.to_timedelta(freq)


def add_savings_stacked_area(aggregate_by, start_date, end_date, fig, row, col):
    freq = AGGREGATE_BY_TO_FREQ[aggregate_by]
    savings = st.session_state.modified_budget["savings"].copy()
    zeros_df = pd.DataFrame(
        [{"date": start_date, "category": category,
          "amount": 0, "converted_amount": 0}
         for category in st.session_state.modified_budget["init"].loc[1:, "field"].unique()])
    savings = pd.concat([zeros_df, savings])
    savings["converted_amount"] = savings.groupby(
        "category", observed=True)["converted_amount"].cumsum()
    full_date_range = pd.date_range(start=start_date,
                                    end=end_date, freq="1D")
    savings = savings.set_index("date")
    plot_df_init = (savings  # noqa: PD010
               .pivot(columns="category", values="converted_amount")
               .reindex(full_date_range)
               .ffill()
               .fillna(0)
               )

    true_amount = (savings  # noqa: PD010
               .pivot(columns="category", values="amount")
               .reindex(full_date_range)
               .ffill()
               .fillna(0)
               )

    currency = (savings  # noqa: PD010
               .pivot(columns="category", values="currency")
               .reindex(full_date_range)
               .ffill()
               )

    for column in plot_df_init.columns:
        if any(st.session_state.modified_budget["init"]["field"] == column):
            init_saving = st.session_state.modified_budget["init"].loc[
                st.session_state.modified_budget["init"]["field"] == column,
                "converted_amount"].to_numpy()
            for i in init_saving:
                plot_df_init[column] = plot_df_init[column] + i

            init_saving = st.session_state.modified_budget["init"].loc[
                st.session_state.modified_budget["init"]["field"] == column,
                "amount"].to_numpy()
            for i in init_saving:
                true_amount[column] = true_amount[column] + i

            init_currency = st.session_state.modified_budget["init"].loc[
                st.session_state.modified_budget["init"]["field"] == column,
                "currency"].to_numpy()
            for i in init_currency:
                if i not in currency[column].cat.categories:
                    currency[column] = currency[column].cat.add_categories(i)
                currency[column] = currency[column].fillna(i)

    plot_df_init = plot_df_init.loc[:, plot_df_init.sum().sort_values(ascending=True).index]
    true_amount = true_amount.loc[:, plot_df_init.sum().sort_values(ascending=True).index]
    currency = currency.loc[:, plot_df_init.sum().sort_values(ascending=True).index].bfill()

    plot_df_init = plot_df_init.resample(freq).last()
    true_amount = true_amount.resample(freq).last()
    currency = currency.resample(freq).last()

    plot_df = plot_df_init.cumsum(axis=1)

    for c, column in enumerate(plot_df.columns):
        hovertemplate = (
            ("<b>{{column}}</b><br>"
             "Date: %{x}<br>"
             "Amount: %{customdata[0]} {{currency}} "
             "(%{customdata[1]} %{customdata[2]})<br>"
             "Sum: %{y} {{currency}}"
             "<extra></extra>")
            if currency[column].iloc[-1] != st.session_state.target_currency
            else (
                "<b>{{column}}</b><br>"
                "Date: %{x}<br>"
                "Amount: %{customdata[0]} {{currency}}<br>"
                "Sum: %{y} {{currency}}"
                "<extra></extra>"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plot_df.index,
                y=(plot_df[column] if st.session_state.target_currency == "BTC"
                   else round(plot_df[column], 2)),
                name=column,
                mode="none" if len(plot_df) > 1 else "markers",
                marker_color=pio.templates[pio.templates.default].layout.colorway[c],
                fillcolor=f"rgba{
                    (*color_to_rgb(pio.templates[pio.templates.default].layout.colorway[c]), 0.5)}",
                line_shape="spline",
                hovertemplate=(
                    hovertemplate
                    .replace("{{column}}", column)
                    .replace("{{currency}}", st.session_state.target_currency)
                ),
                customdata=np.stack(
                    [plot_df_init[column].to_numpy() if st.session_state.target_currency == "BTC"
                     else plot_df_init[column].to_numpy().round(2),
                     true_amount[column].to_numpy(),
                     currency[column].to_numpy()], axis=-1),
                line_smoothing=.9,
                fill="tonexty",
                showlegend=False,
            ),
            row=row, col=col)
    return fig


def trends_plot(aggregate_by, show_fixed):
    aggregate_by_to_adj = {
        "days": "Daily",
        "weeks": "Weekly",
        "months": "Monthly",
    }

    st.markdown(f"##### {aggregate_by_to_adj[aggregate_by]} Trends")

    fig = make_subplots(rows=3, cols=1,
                        vertical_spacing=0.15,
                        subplot_titles=("Expenses", "Balance", "Savings"),
                        shared_xaxes="all")

    fig, start_date, end_date = add_balance_lineplot(aggregate_by, fig, row=2, col=1)
    fig = add_expenses_ridgeline(show_fixed, aggregate_by, start_date, end_date,
                                 fig=fig, row=1, col=1)
    fig = add_savings_stacked_area(aggregate_by, start_date, end_date,
                                   fig=fig, row=3, col=1)

    fig.update_layout(template="one_light",
                      plot_bgcolor="#fafafa",
                      dragmode="pan",
                      margin={"l": 110, "r": 0, "t": 30, "b": 50},
                      height=700,
                      xaxis_showticklabels=True,
                      xaxis2_showticklabels=True,
                      yaxis2_showticklabels=True,
                      xaxis2_showgrid=True,
                      yaxis2_showgrid=True,
                      yaxis2_title=st.session_state.target_currency,
                      yaxis3_showticklabels=True,
                      xaxis3_showgrid=True,
                      yaxis3_showgrid=True,
                      yaxis3_title=st.session_state.target_currency,
                      showlegend=False,
                      legend=dict(x=0.017,
                                  y=0.62,
                                  bgcolor="rgba(250, 250, 250, 0.8)",
                                  xanchor="left",
                                  yanchor="top",
                                  tracegroupgap=0,
                                  traceorder="reversed"),
                      )

    fig.update_annotations(font=dict(size=18), yshift=10)
    fig.update_annotations(selector=dict(name="ytitles"), font=dict(size=14), xshift=-10)
    fig.update_xaxes(title=None, tickfont=dict(size=14))
    fig.update_yaxes(showline=False, tickfont=dict(size=14))

    st.plotly_chart(fig, theme=None)


def restart_button():
    if st.button("⟳", help="Clear cache and restart"):
        st.session_state.first_run = True
        st.cache_data.clear()
        st.rerun()


def extract_sheet_id(url):
    match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
    return match.group(1) if match else None


@st.dialog("Input Google Sheets link")
def request_google_doc_link():
    link = st.text_input("Link to the Google Sheets with budget")
    st.markdown("""
The easiest way to create a valid Google Sheet for the app is to populate
[this template](https://docs.google.com/spreadsheets/d/19wTUH2nv4bkI2fPUgGaLDiFV9tyY3N1j3hBoVx2zBsk)
with your values.


### How to start using this app
1. Create an editable copy of the template and save it to your Google Drve.
2. Ensure that the sharing mode of the sheet is "Everyone can view".
3. Input your values in the template.
        """, unsafe_allow_html=True)

    if link:
        st.session_state.google_doc_id = extract_sheet_id(link)
        if st.session_state.google_doc_id:
            st.success("Google Sheets link is valid!")
        else:
            st.error("Invalid Google Sheets link. Please try again.")
    if st.session_state.google_doc_id:
        st.rerun()


def main():
    customize_page_appearance()
    initialize_session_state()
    if st.session_state.google_doc_id is None:
        request_google_doc_link()

    if st.session_state.google_doc_id:
        try:
            st.session_state.budget = get_data(st.session_state.google_doc_id)
        except urllib.error.HTTPError as e:
            st.error(f"Google Sheets document probably has restricted access. Got an error: {e}\n\n"
                     "Reload page and try again.")
            st.stop()

        st.session_state.exchange_rates = get_exchange_rates()
        gmt_plus_4 = datetime.timezone(datetime.timedelta(hours=4))
        now_gmt4 = datetime.datetime.now(gmt_plus_4)

        st.title("Rich Secret 💰", anchor=False)

        col1, col2 = st.columns([20, 1])
        col1.subheader("Settings", anchor=False)
        with col2:
            restart_button()

        st.session_state.target_currency, start_date, end_date, aggregate_by = settings(
            st.session_state.budget, now_gmt4)

        st.session_state.modified_budget = convert_amounts_to_target_currency(
            filter_sheets_by_date_range(st.session_state.budget, start_date, end_date),
            st.session_state.target_currency,
            st.session_state.exchange_rates)

        # no idea why I need to rerun it here but otherwise plots don't show
        if st.session_state.first_run:
            st.session_state.first_run = False
            st.rerun()

        st.subheader("Stats", anchor=False)
        stats(aggregate_by, start_date, end_date)

        st.markdown("")
        col1, col2 = st.columns([4, 1], vertical_alignment="bottom")
        col1.subheader("Plots", anchor=False)
        show_fixed = col2.checkbox("Show fixed expenses")

        actual_vs_planned_plot(show_fixed)
        st.markdown("")
        st.markdown("")
        trends_plot(aggregate_by, show_fixed)


if __name__ == "__main__":
    main()
