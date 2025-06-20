#!/bin/bash

# Prompt the user for the version
read -p "Enter the version (e.g., 0.0.2): " VERSION

# Check if the version is provided
if [ -z "$VERSION" ]; then
  echo "Error: Version not provided."
  exit 1
fi

# Define image names
BASE_IMAGE_NAME="s1swot"
REGISTRY_IMAGE_NAME="gitlab-registry.ifremer.fr/lops-wave/s1ifr:s1swot"

# Docker commands
echo "Building Docker image: ${BASE_IMAGE_NAME}:${VERSION}"
docker build . -t "${BASE_IMAGE_NAME}:${VERSION}"

echo "Tagging Docker image: ${BASE_IMAGE_NAME}:${VERSION} -> ${REGISTRY_IMAGE_NAME}-${VERSION}"
docker tag "${BASE_IMAGE_NAME}:${VERSION}" "${REGISTRY_IMAGE_NAME}-${VERSION}"

echo "Pushing Docker image: ${REGISTRY_IMAGE_NAME}-${VERSION}"
docker push "${REGISTRY_IMAGE_NAME}-${VERSION}"

echo "Release process completed for version ${VERSION}."
