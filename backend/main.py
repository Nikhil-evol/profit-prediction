"""
FastAPI application for profit prediction with SHAP explainability.

Prediction endpoint is unchanged in signature and behaviour.
SHAP explanations are derived from the GradientBoostingRegressor inside
the VotingRegressor ensemble so that TreeExplainer can work efficiently.
"""

from __future__ import annotations

import os
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from .feature_engineering import feature_extraction
from .schema import PredictionRequest

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "ensemble_pipeline.pkl"
)

# ---------------------------------------------------------------------------
# Application state (loaded once at startup)
# ---------------------------------------------------------------------------
@dataclass
class AppState:
    pipeline: object          # full sklearn Pipeline
    explainer: shap.TreeExplainer
    feature_names: list[str]


_state: AppState | None = None


def _load_state() -> AppState:
    """Load the pipeline and build the SHAP explainer once at startup."""
    logger.info("Loading model from %s", MODEL_PATH)
    pipeline = joblib.load(MODEL_PATH)

    # Extract sub-objects from the pipeline --------------------------------
    preprocessor = pipeline.named_steps["preprocessor"]
    voting_model  = pipeline.named_steps["model"]

    # GradientBoosting is the explainability surrogate inside the ensemble.
    gb_model = voting_model.named_estimators_["gb"]

    # Build a TreeExplainer once (expensive) --------------------------------
    logger.info("Building SHAP TreeExplainer …")
    explainer = shap.TreeExplainer(gb_model)

    # Feature names after preprocessing ------------------------------------
    feature_names: list[str] = list(preprocessor.get_feature_names_out())

    logger.info("Startup complete – %d features", len(feature_names))
    return AppState(pipeline=pipeline, explainer=explainer, feature_names=feature_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    _state = _load_state()
    yield
    # Nothing to tear down, but the hook is here for future use.


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Profit Prediction API",
    description="Predicts order-level profit and explains the prediction via SHAP.",
    version="2.0.0",
    lifespan=lifespan,
)

@app.get("/")
def home():
    return {
        "project": "Global Superstore Profit Prediction API",
        "status": "running"
    }


def _build_input_df(
    sales: float,
    discount: float,
    quantity: int,
    shipping_cost: float,
    category: str,
    segment: str,
    sub_category: str,
    region: str,
    order_date: date,
) -> pd.DataFrame:
    """Construct the raw feature DataFrame for one order."""



    return feature_extraction(
        pd.DataFrame({
            "Sales":         [sales],
            "Discount":      [discount],
            "Quantity":      [quantity],
            "Shipping Cost": [shipping_cost],
            "Category":      [category],
            "Segment":       [segment],
            "Sub-Category":  [sub_category],
            "Region":        [region],
            "Year":          [order_date.year],
            "weeknum":       [order_date.isocalendar()[1]],
        })
    )


# ---------------------------------------------------------------------------
# SHAP helper
# ---------------------------------------------------------------------------
TOP_N = 3


def _compute_shap_explanation(
    df: pd.DataFrame,
) -> dict:
    """
    Transform *df* with the pipeline preprocessor, compute SHAP values using
    the GradientBoosting sub-model, and return the top-N positive / negative
    feature drivers.

    Returns
    -------
    dict with keys ``positive_drivers`` and ``negative_drivers``, each a list
    of ``{"feature": str, "shap_value": float}`` dicts sorted by |SHAP value|.
    """
    assert _state is not None, "App state not initialised"

    preprocessor  = _state.pipeline.named_steps["preprocessor"]
    explainer     = _state.explainer
    feature_names = _state.feature_names
    


    # Transform input with the same preprocessor the pipeline uses ----------
    X_transformed = preprocessor.transform(df)
    

    # SHAP values for a single row: shape (1, n_features) ------------------
    shap_values = explainer.shap_values(X_transformed)

    
    # For GradientBoostingRegressor shap_values returns a 2-D array
    row_shap: np.ndarray = np.asarray(shap_values[0])  # shape (n_features,)

    # Pair each feature with its SHAP value --------------------------------
    pairs = list(zip(feature_names, row_shap.tolist()))
    

    positive = sorted(
        [(name, float(val)) for name, val in pairs if val > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:TOP_N]

    negative = sorted(
        [(name, float(val)) for name, val in pairs if val < 0],
        key=lambda x: x[1],
    )[:TOP_N]

    FEATURE_MAPPING = {
    "Discount": "Discount",
    "log_sales": "Sales (Log Scale)",
    "Shipping Cost": "Shipping Cost",
    "discount_amount": "Discount Amount",
    "shipping_cost_ratio": "Shipping Cost Ratio",
    "high_discount": "High Discount Flag",
    "Region_West": "Region: West",
    "Sub-Category_Copiers": "Sub-Category: Copiers",
    "discount_bucket_0_none": "Discount Level",
    "discount_bucket_1_low": "Discount Level",
    "discount_bucket_2_medium": "Discount Level",
    "discount_bucket_3_high": "Discount Level",
    "discount_bucket_4_extreme": "Discount Level",
    "has_discount": "Discount Applied",
    "high_discount": "High Discount (>30%)",
    "log_sales": "Sales (Log Scale)"
}
    

    return {
    "positive_drivers": [
        {
            "feature": FEATURE_MAPPING.get(
                name.replace("num__", "").replace("cat__", ""),
                name.replace("num__", "").replace("cat__", "")
            ),
            "shap_value": round(val, 4),
            "effect": "increased predicted profit"
        }
        for name, val in positive
    ],
    "negative_drivers": [
        {
            "feature": FEATURE_MAPPING.get(
                name.replace("num__", "").replace("cat__", ""),
                name.replace("num__", "").replace("cat__", "")
            ),
            "shap_value": round(val, 4),
            "effect": "decreased predicted profit"
        }
        for name, val in negative
    ],
}



class FeatureContribution(BaseModel):
    feature: str
    shap_value: float = Field(..., description="SHAP contribution to predicted profit (USD)")


class ShapExplanation(BaseModel):
    positive_drivers: list[FeatureContribution] = Field(
        ...,
        description="Top features that increased the predicted profit, ranked by impact",
    )
    negative_drivers: list[FeatureContribution] = Field(
        ...,
        description="Top features that decreased the predicted profit, ranked by impact",
    )


class PredictionResponse(BaseModel):
    predicted_profit: float = Field(..., description="Predicted profit in USD")
    status: str             = Field(..., description="'Profit' or 'Loss'")
    message: str            = Field(..., description="Human-readable summary")
    shap_explanation: ShapExplanation = Field(
        ...,
        description=(
            "SHAP-based explanation derived from the GradientBoosting component "
            "of the ensemble. Indicates which features drove the prediction up or down."
        ),
    )


# ---------------------------------------------------------------------------
# Core prediction + explanation logic
# ---------------------------------------------------------------------------
def _predict_with_explanation(req: PredictionRequest) -> PredictionResponse:
    assert _state is not None

    df = _build_input_df(
        sales=req.sales,
        discount=req.discount,
        quantity=req.quantity,
        shipping_cost=req.shipping_cost,
        category=req.category,
        segment=req.segment,
        sub_category=req.sub_category,
        region=req.region,
        order_date=req.order_date,
    )

    # Full-ensemble prediction (unchanged behaviour) -----------------------
    raw_prediction = _state.pipeline.predict(df)[0]
    # Handle both scalar and array-like returns defensively
    predicted_profit = float(
        raw_prediction[0] if hasattr(raw_prediction, "__len__") else raw_prediction
    )

    # SHAP explanation via GradientBoosting --------------------------------
    shap_data = _compute_shap_explanation(df)

    # Status & message ------------------------------------------------------
    if predicted_profit >= 0:
        status  = "Profit"
        message = "This order is expected to be profitable."
    else:
        status  = "Loss"
        message = "You may lose money on this order."

    return PredictionResponse(
        predicted_profit=round(predicted_profit, 4),
        status=status,
        message=message,
        shap_explanation=ShapExplanation(
            positive_drivers=[FeatureContribution(**d) for d in shap_data["positive_drivers"]],
            negative_drivers=[FeatureContribution(**d) for d in shap_data["negative_drivers"]],
        ),
    )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict order profit with SHAP explanation",
    tags=["Prediction"],
)
def predict(req: PredictionRequest) -> PredictionResponse:
    """
    Predict the profit for a single order **and** return a SHAP-based
    explanation of which features drove the result.

    The SHAP values are computed using the **GradientBoosting** component
    inside the VotingRegressor ensemble, making TreeExplainer both accurate
    and efficient.  Positive ``shap_value`` means the feature *increased*
    the predicted profit; negative means it *decreased* it.
    """
    try:
        return _predict_with_explanation(req)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Ops"])
def health():
    return {"status": "ok"}