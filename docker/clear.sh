#!/usr/bin/env bash
set -euo pipefail

echo "Stopping running containers..."
running_containers="$(docker ps -q)"
if [ -n "$running_containers" ]; then
  docker kill $running_containers
else
  echo "No running containers."
fi

echo "Removing all containers..."
all_containers="$(docker ps -aq)"
if [ -n "$all_containers" ]; then
  docker rm -f $all_containers
else
  echo "No containers to remove."
fi

echo "Pruning Docker system..."
docker system prune -a --volumes -f

echo "Pruning builder cache..."
docker builder prune -a -f

echo "Done."