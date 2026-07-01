"""Module 3 — Budget/Spend Forecasting.

Designed for SHORT series (12–36 months) typical of small entities.

Strategy
--------
1. Try Prophet first (handles seasonality, holiday effects, works on ~12 pts).
2. Fall back to ARIMA via statsmodels if Prophet unavailable.
3. Always produce a simple Holt-Winters ETS as a sanity-check baseline.

Output per category: 6–12 month forecast + 80/95% confidence intervals
+ a budget-breach flag when the trajectory exceeds user-supplied budget.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    category: str
    forecast_df: pd.DataFrame        # columns: ds, yhat, yhat_lower, yhat_upper
    budget: Optional[float]
    breach_month: Optional[str]      # first month forecast exceeds budget
    breach_amount: Optional[float]
    plain_english: str


def _try_prophet(history: pd.DataFrame, periods: int, freq: str) -> Optional[pd.DataFrame]:
    """history must have columns ds (datetime) and y (spend)."""
    try:
        from prophet import Prophet
    except ImportError:
        return None
    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=len(history) >= 24,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.15,
    )
    m.fit(history, iter=300)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    fc = m.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)


def _try_arima(history: pd.DataFrame, periods: int) -> Optional[pd.DataFrame]:
    try:
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        return None

    y = history["y"].values
    # Auto-select differencing order via ADF test
    d = 0
    if len(y) >= 10:
        p_val = adfuller(y)[1]
        if p_val > 0.05:
            d = 1
    try:
        model = ARIMA(y, order=(2, d, 1))
        fit = model.fit()
        fc = fit.get_forecast(steps=periods)
        ci = fc.conf_int(alpha=0.2)  # 80% CI
        last_date = history["ds"].max()
        future_dates = pd.date_range(last_date, periods=periods + 1, freq="MS")[1:]
        return pd.DataFrame({
            "ds": future_dates,
            "yhat": fc.predicted_mean,
            "yhat_lower": ci.iloc[:, 0].values,
            "yhat_upper": ci.iloc[:, 1].values,
        })
    except Exception as e:
        logger.warning("ARIMA failed: %s", e)
        return None


def _holt_winters(history: pd.DataFrame, periods: int) -> pd.DataFrame:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    y = history["y"].values
    seasonal = "add" if len(y) >= 24 else None
    sp = 12 if seasonal else None
    try:
        model = ExponentialSmoothing(y, trend="add", seasonal=seasonal, seasonal_periods=sp)
        fit = model.fit()
        yhat = fit.forecast(periods)
    except Exception:
        # Naive: repeat last value with slight trend
        trend = np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 3 else 0
        yhat = y[-1] + trend * np.arange(1, periods + 1)

    last_date = history["ds"].max()
    future_dates = pd.date_range(last_date, periods=periods + 1, freq="MS")[1:]
    std = float(np.std(y)) if len(y) > 1 else float(y[-1]) * 0.1
    return pd.DataFrame({
        "ds": future_dates,
        "yhat": yhat,
        "yhat_lower": yhat - 1.28 * std,
        "yhat_upper": yhat + 1.28 * std,
    })


class SpendForecaster:
    """Forecast monthly category spend with budget breach detection."""

    def __init__(self, periods: int = 12, freq: str = "MS"):
        self.periods = periods
        self.freq = freq

    def forecast(
        self,
        df: pd.DataFrame,
        date_col: str = "month",
        amount_col: str = "amount",
        category_col: Optional[str] = "category",
        budgets: Optional[dict[str, float]] = None,
    ) -> list[ForecastResult]:
        """Forecast each category (or total if no category_col).

        df should have monthly rows. Minimum 6 data points recommended.
        """
        budgets = budgets or {}
        results = []

        if category_col and category_col in df.columns:
            groups = df.groupby(category_col)
        else:
            groups = [("All Spend", df)]

        for cat, group in groups:
            history = (
                group.groupby(date_col)[amount_col]
                .sum()
                .reset_index()
                .rename(columns={date_col: "ds", amount_col: "y"})
                .sort_values("ds")
            )
            history["ds"] = pd.to_datetime(history["ds"])

            if len(history) < 3:
                logger.warning("Category '%s' has < 3 data points; skipping.", cat)
                continue

            fc = _try_prophet(history, self.periods, self.freq)
            if fc is None:
                fc = _try_arima(history, self.periods)
            if fc is None:
                fc = _holt_winters(history, self.periods)
                logger.info("Using Holt-Winters for '%s'", cat)
            else:
                logger.info("Using Prophet/ARIMA for '%s'", cat)

            budget = budgets.get(str(cat))
            breach_month = breach_amount = None
            if budget is not None:
                over = fc[fc["yhat"] > budget]
                if not over.empty:
                    breach_month = str(over["ds"].iloc[0])[:7]
                    breach_amount = round(float(over["yhat"].iloc[0]) - budget, 2)

            avg_hist = float(history["y"].mean())
            avg_fc = float(fc["yhat"].mean())
            trend_pct = (avg_fc - avg_hist) / (avg_hist + 1e-6) * 100
            trend_str = (
                f"spending is projected to {'increase' if trend_pct > 0 else 'decrease'} "
                f"{abs(trend_pct):.1f}% on average over the next {self.periods} months"
            )
            plain = f"{cat}: {trend_str}."
            if breach_month:
                plain += (
                    f" Budget of ${budget:,.0f}/month is forecast to be exceeded "
                    f"starting {breach_month} by ${breach_amount:,.0f}."
                )

            results.append(ForecastResult(
                category=str(cat),
                forecast_df=fc.reset_index(drop=True),
                budget=budget,
                breach_month=breach_month,
                breach_amount=breach_amount,
                plain_english=plain,
            ))

        return results
