from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .constants import (
    CATEGORIES,
    REGIONS,
    SEGMENTS,
    SUB_CATEGORIES,
    ALL_SUB_CATEGORIES
)


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    All fields required for profit prediction.
    Categorical strings are validated against the allowed lists in constants.py.
    """

    sales: float = Field(
        ...,
        gt=0,
        description="Total sales amount in USD. Must be greater than 0.",
        examples=[500.0],
    )
    discount: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Discount fraction applied to the order (0 = no discount, 1 = 100% off).",
        examples=[0.2],
    )
    quantity: int = Field(
        ...,
        gt=0,
        description="Number of units ordered. Must be a positive integer.",
        examples=[3],
    )
    shipping_cost: float = Field(
        ...,
        ge=0.0,
        description="Shipping cost in USD. Zero is allowed (e.g. free shipping).",
        examples=[14.5],
    )
    category: str = Field(
        ...,
        description=f"Product category. Allowed values: {CATEGORIES}",
        examples=["Technology"],
    )
    segment: str = Field(
        ...,
        description=f"Customer segment. Allowed values: {SEGMENTS}",
        examples=["Consumer"],
    )
    sub_category: str = Field(
        ...,
        description=f"Product sub-category. Allowed values: {ALL_SUB_CATEGORIES}",
        examples=["Phones"],
    )
    region: str = Field(
        ...,
        description=f"Geographic region. Allowed values: {REGIONS}",
        examples=["West"],
    )
    order_date: date = Field(
        ...,
        description="Date the order was placed (YYYY-MM-DD).",
        examples=["2024-06-15"],
    )

    model_config = {"populate_by_name": True}

    # ── Categorical validators ────────────────────────────────────────────────

    @field_validator("sub_category")
    @classmethod
    def validate_sub_category(cls, v: str) -> str:
        if v not in ALL_SUB_CATEGORIES:
            raise ValueError(
            f"Invalid sub_category '{v}'. Allowed values: {ALL_SUB_CATEGORIES}"
           )
        return v
    
    #if subcategeory not in that categeory inside then error
    @model_validator(mode="after")
    def validate_category_subcategory(self):

        if self.sub_category not in SUB_CATEGORIES[self.category]:
            raise ValueError(
                f"'{self.sub_category}' is not valid for category '{self.category}'"
            )

        return self

    @field_validator("segment")
    @classmethod
    def validate_segment(cls, v: str) -> str:
        if v not in SEGMENTS:
            raise ValueError(
                f"Invalid segment '{v}'. Allowed values: {SEGMENTS}"
            )
        return v


    @field_validator("region")
    @classmethod
    def validate_region(cls, v: str) -> str:
        if v not in REGIONS:
            raise ValueError(
                f"Invalid region '{v}'. Allowed values: {REGIONS}"
            )
        return v

    # ── Cross-field guard ─────────────────────────────────────────────────────

    @model_validator(mode="after")
    def validate_order_date(self):

        if self.order_date.year < 2010:
            raise ValueError("Order date is too old.")

        if self.order_date.year > 2030:
            raise ValueError("Order date is unrealistic.")

        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "sales": 500.0,
                "discount": 0.2,
                "quantity": 3,
                "shipping_cost": 14.5,
                "category": "Technology",
                "segment": "Consumer",
                "sub_category": "Phones",
                "region": "West",
                "order_date": "2026-06-15",
            }
        }
    }




# ── Error shape (used in openapi responses dict) ──────────────────────────────

class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Invalid category 'Electronics'."])