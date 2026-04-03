docker build -t dashllipsey/songbird_backend:latest -f backend.dockerfile ../..
docker build -t dashllipsey/songbird_frontend:latest -f frontend.dockerfile ../..
docker push dashllipsey/songbird_backend:latest
docker push dashllipsey/songbird_frontend:latest