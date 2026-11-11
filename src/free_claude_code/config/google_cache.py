"""Explicit price inputs for hypothetical Google context-cache comparisons."""

import json
from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class GoogleCachePrice(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    provider: Literal["vertex"]
    model: str = Field(min_length=1)
    input_per_million: float = Field(ge=0)
    cached_input_per_million: float = Field(ge=0)
    storage_per_million_hour: float = Field(ge=0)
    output_per_million: float | None = Field(default=None, ge=0)
    usd_to_brl: float | None = Field(default=None, gt=0)
    usd_to_brl_verified_at: date | None = None
    fx_source: HttpUrl | None = None
    price_region: Literal["global", "non_global"] = "global"
    minimum_tokens: int = Field(gt=0)
    ttl_seconds: int = Field(default=300, ge=60)
    verified_at: date
    valid_until: date
    source: HttpUrl

    @model_validator(mode="after")
    def validate_source_and_dates(self):
        if self.valid_until < self.verified_at:
            raise ValueError("price validity must end after verification")
        if self.source.host not in {"ai.google.dev", "cloud.google.com", "docs.cloud.google.com"}:
            raise ValueError("cache prices require an official Google source")
        return self


def decode_google_cache_prices(value):
    return json.loads(value) if isinstance(value, str) else value
