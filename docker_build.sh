#!/bin/bash
# Clean build script
export DOCKER_HOST="unix:///mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/docker.sock"
cd "$(dirname "$0")"
exec docker build -t rag-demo:latest --progress plain . 2>&1
