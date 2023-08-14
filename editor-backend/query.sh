#!/bin/sh

curl http://localhost:5000/

curl -X POST -H "Content-Type: application/json" -d '{
  "prompt": "The child who was playing in the park was having a lot of fun, and he was not paying attention to anything else"
}' http://localhost:5000/predict