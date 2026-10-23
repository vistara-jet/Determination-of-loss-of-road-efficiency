# Determination of Loss of Road Efficiency

Image-based road capacity analysis for Indian urban roads. YOLOv8 detects obstructions (potholes, illegal parking, street vendors, carts, garbage, barricades), and the capacity engine converts them into capacity loss, V/C ratio and Level of Service using IRC codes.

- **App:** `road_analyzer_deploy/` (FastAPI + web dashboard, Docker/Render ready)
- **Method and formulas:** [`road_analyzer_deploy/METHODOLOGY.md`](road_analyzer_deploy/METHODOLOGY.md)
- **Standards:** IRC:106-1990, IRC:106 (2022 draft revision), IRC:82-2015

## Run locally (Python 3.11)

```bash
cd road_analyzer_deploy
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn road_analyzer.app:app --reload
```
Then open http://localhost:8000.

## Deploy on Render

New → Web Service → connect this repo → set **Root Directory** to `road_analyzer_deploy` (Docker runtime is detected from `render.yaml`).

Based on [ERpdeore/Indian-Road-Capacity-Analyzer-Digital-Twin](https://github.com/ERpdeore/Indian-Road-Capacity-Analyzer-Digital-Twin).
