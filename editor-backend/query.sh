#!/bin/sh

curl http://localhost:5000/

curl -X POST -H "Content-Type: application/json" -d '{
  "prompt": "King Edward, be it remembered, was a man of many and varied interests"
}' http://localhost:5000/predict

curl http://localhost:5000/save

curl -X POST -H "Content-Type: application/json" -d '{
  "annotations": [{"chosen":"true", "model":"something", "prompt":"something"}, {"model":"something", "prompt":"something"}]
}' http://localhost:5000/annotate