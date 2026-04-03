docker build -t dashllipsey/backend:latest -f backend.dockerfile ../..
docker build -t dashllipsey/frontend:latest -f frontend.dockerfile ../..
docker push dashllipsey/backend:latest
docker push dashllipsey/frontend:latest