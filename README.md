# Virtual Diabetes Clinic - Risk Prediction Service (Group-ZZ)

This is an MLOps assignment to build a service that predicts disease progression risk for diabetic patients.

## v0.2 (Latest)
- **Model**: `StandardScaler` + `RandomForestRegressor`
- **RMSE**: 54.398350726816645
- **Model Version**: `v0.2`

## Quick Start

### Pull from GitHub Container Registry (Recommended)
```bash
docker pull ghcr.io/firrrdragon/group-zz:0.21
docker run -d -p 9696:9696 ghcr.io/firrrdragon/group-zz:0.21
```

### Build Locally
```bash
docker build -t diabetes-service:v0.2 .
docker run -d -p 9696:9696 diabetes-service:v0.2
```


## How to Run (using Docker)

This project is containerized with Docker, and the model is baked into the image.

1.  **Build the v0.2 image:**
    ```bash
    docker build -t diabetes-service:v0.2 .
    ```

2.  **Run the v0.2 container:**
    ```bash
    docker run -d -p 9696:9696 --name diabetes_v02 diabetes-service:v0.2
    ```

## API Endpoints

### GET /health
Checks the service health and model version.

```bash
curl http://localhost:9696/health

#Response:

#JSON

{
  "model_version": "0.2",
  "status": "ok"
}
POST /predict
Sends patient data to get a risk score prediction.

#Example Request:

Bash

curl -X POST http://localhost:9696/predict \
     -H "Content-Type: application/json" \
     -d '{ "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001 }'
#Response:

JSON

{
  "prediction": 156.4... 
}
