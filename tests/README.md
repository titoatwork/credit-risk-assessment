# Tests

| File | What it verifies |
|------|------------------|
| `test_train.py` | Both models train on real Home Credit sample; ROC-AUC written |
| `test_predict.py` | Decision + probability; threshold rule consistency |
| `test_explain.py` | SHAP top features for a declined case |
| `test_api.py` | FastAPI `/health`, `/assess`, `/explain`, validation errors |

Run from project root with the project venv active:

```powershell
pytest -q
```
