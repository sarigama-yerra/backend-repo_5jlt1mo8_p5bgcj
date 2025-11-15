import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal, Optional
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Utility functions ----------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF using error function (no external deps)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(
    S: float, K: float, r: float, sigma: float, T: float, option_type: Literal["call", "put"]
) -> float:
    if S <= 0 or K <= 0 or sigma <= 0 or T <= 0:
        raise ValueError("S, K, sigma, and T must be positive")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bond_price(face: float, coupon_rate: float, ytm: float, years: float, freq: int = 2) -> float:
    """Price a fixed-coupon bond with flat yield (ytm, nominal comp)."""
    if freq <= 0 or years <= 0:
        raise ValueError("freq and years must be positive")
    periods = int(round(years * freq))
    c = face * coupon_rate / freq
    y = ytm / freq
    price = 0.0
    for t in range(1, periods + 1):
        price += c / ((1 + y) ** t)
    price += face / ((1 + y) ** periods)
    return price


def swap_pv(
    notional: float,
    fixed_rate: float,
    pay_fixed: bool,
    years: float,
    freq: int,
    flat_rate: float,
) -> float:
    """
    Very simple plain-vanilla interest rate swap PV using a flat curve.
    Positive PV is value to the receiver of the fixed leg.
    If pay_fixed=True, PV to the fixed payer is negative of receive-fixed PV.
    """
    if freq <= 0 or years <= 0:
        raise ValueError("freq and years must be positive")
    periods = int(round(years * freq))
    r = flat_rate / freq
    # Discount factors
    dfs = [(1 / ((1 + r) ** t)) for t in range(1, periods + 1)]
    # Annuity (sum of DFs)
    annuity = sum(dfs)
    # Par rate implied by this flat curve
    par_rate = (1 - dfs[-1] / (1 + r))  # incorrect; let's compute via formula using bond math
    # Correct par rate: fixed rate that sets PV of fixed = PV of floating = 1 - DF(T)
    # PV floating (receive) at inception equals 1 - DF(T)
    df_T = (1 / ((1 + r) ** periods))
    pv_float = 1 - df_T
    # Fixed leg PV per unit notional at rate R: R/freq * annuity + df_T (for principal exchange in bond analogy) but swap has no principal exchange.
    # For swap, fixed leg PV (per notional) = fixed_rate/freq * annuity
    pv_fixed_per_unit = fixed_rate / freq * annuity
    pv_receive_fixed = notional * (pv_fixed_per_unit - pv_float)
    return -pv_receive_fixed if pay_fixed else pv_receive_fixed

# ---------- Pydantic models ----------

class OptionRequest(BaseModel):
    S: float = Field(..., description="Spot price")
    K: float = Field(..., description="Strike price")
    r: float = Field(..., description="Risk-free rate (annual, continuously compounded not required)")
    sigma: float = Field(..., description="Volatility (annualized)")
    T: float = Field(..., description="Time to maturity in years")
    option_type: Literal["call", "put"] = "call"

class BondRequest(BaseModel):
    face: float = 1000.0
    coupon_rate: float = Field(..., description="Annual coupon rate, e.g., 0.05 for 5%")
    ytm: float = Field(..., description="Annual yield to maturity, nominal comp")
    years: float = Field(..., description="Years to maturity")
    freq: int = 2

class SwapRequest(BaseModel):
    notional: float = 1_000_000
    fixed_rate: float = Field(..., description="Annual fixed rate (e.g., 0.03)")
    pay_fixed: bool = Field(True, description="If true, value to fixed payer; else to fixed receiver")
    years: float = Field(..., description="Swap maturity in years")
    freq: int = 4
    flat_rate: float = Field(..., description="Flat annual rate for discounting and float projection")

# ---------- Routes ----------

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.post("/api/price/option")
def price_option(req: OptionRequest):
    try:
        price = black_scholes_price(req.S, req.K, req.r, req.sigma, req.T, req.option_type)
        return {"instrument": "option", "input": req.dict(), "price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/price/bond")
def price_bond(req: BondRequest):
    try:
        price = bond_price(req.face, req.coupon_rate, req.ytm, req.years, req.freq)
        return {"instrument": "bond", "input": req.dict(), "price": price}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/price/swap")
def price_swap(req: SwapRequest):
    try:
        pv = swap_pv(req.notional, req.fixed_rate, req.pay_fixed, req.years, req.freq, req.flat_rate)
        return {"instrument": "swap", "input": req.dict(), "pv": pv}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
