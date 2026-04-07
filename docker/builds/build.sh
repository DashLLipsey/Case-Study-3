docker build -t dashlipsey/songbird_backend:latest -f backend.dockerfile ../..
docker build -t dashlipsey/songbird_frontend:latest -f frontend.dockerfile ../..
docker push dashlipsey/songbird_backend:latest
docker push dashlipsey/songbird_frontend:latest