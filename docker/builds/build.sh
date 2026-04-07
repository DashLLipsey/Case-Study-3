docker build -t group03/songbird_backend:latest -f backend.dockerfile ../..
docker build -t group03/songbird_frontend:latest -f frontend.dockerfile ../..
docker push group03/songbird_backend:latest
docker push group03/songbird_frontend:latest