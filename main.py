from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np
from pathlib import Path
from config.paths_config import MODEL_OUTPUT_PATH

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Load model - replace with your actual model path
try:
    model = joblib.load(MODEL_OUTPUT_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("indexx.html", {"request": request, "prediction": None})

@app.post("/", response_class=HTMLResponse)
async def make_prediction(
    request: Request,
    lead_time: int = Form(...),
    no_of_special_request: int = Form(...),
    avg_price_per_room: float = Form(...),
    arrival_month: int = Form(...),
    arrival_date: int = Form(...),
    market_segment_type: int = Form(...),
    no_of_week_nights: int = Form(...),
    no_of_weekend_nights: int = Form(...),
    type_of_meal_plan: int = Form(...),
    room_type_reserved: int = Form(...)
):
    if model is None:
        return templates.TemplateResponse("indexx.html", {
            "request": request,
            "error": "Model not loaded"
        })
    
    try:
        features = np.array([[
            lead_time, no_of_special_request, avg_price_per_room,
            arrival_month, arrival_date, market_segment_type,
            no_of_week_nights, no_of_weekend_nights,
            type_of_meal_plan, room_type_reserved
        ]])

        prediction = model.predict(features)[0]
        
        return templates.TemplateResponse(
            "indexx.html", 
            {
                "request": request,
                "prediction": prediction,
                "lead_time": lead_time,
                "no_of_special_request": no_of_special_request,
                "avg_price_per_room": avg_price_per_room,
                "arrival_month": arrival_month,
                "arrival_date": arrival_date,
                "market_segment_type": market_segment_type,
                "no_of_week_nights": no_of_week_nights,
                "no_of_weekend_nights": no_of_weekend_nights,
                "type_of_meal_plan": type_of_meal_plan,
                "room_type_reserved": room_type_reserved
            }
        )
    except Exception as e:
        return templates.TemplateResponse("indexx.html", {
            "request": request,
            "error": f"Prediction error: {str(e)}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)