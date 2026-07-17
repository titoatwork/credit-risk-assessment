"""
Credit Risk Assessment System
=============================

End-to-end educational package for risk-based credit approve/decline decisions:

* Train logistic regression and XGBoost on Home Credit-style application data
* Preprocess via sklearn Pipelines (impute, scale, one-hot)
* Decide approve/decline from predicted default probability
* Explain declines with SHAP
* Serve via FastAPI (``credit_risk.api:app``)

Public entry points
-------------------
* Training CLI: ``python -m credit_risk.train``
* API: ``uvicorn credit_risk.api:app --host 127.0.0.1 --port 8000``
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
