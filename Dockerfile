# ---- Base Stage ----
# This stage installs micromamba
FROM ubuntu:22.04 as micromamba-base

ARG MAMBA_VERSION=1.5.6
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Install micromamba into /usr/local/bin/
# Using a separate stage allows for better caching
RUN apt-get update && apt-get install -y curl bzip2 ca-certificates && \
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/${MAMBA_VERSION} | \
    tar -xvj --strip-components=1 -C /usr/local/bin/ bin/micromamba && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- Final Stage ----
# This is the image we will actually use
FROM ubuntu:22.04

# Copy micromamba from the base stage
COPY --from=micromamba-base /usr/local/bin/micromamba /usr/local/bin/micromamba

# Set up environment variables
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    ENV_NAME=myenv

# Install system dependencies needed by your Python packages
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libxext6 libsm6 libxrender1 git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy environment file and create the conda environment
COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n $ENV_NAME -f /tmp/environment.yml && \
    micromamba clean -a -y && \
    rm /tmp/environment.yml

# This is the modern way to activate the environment for all subsequent commands
# It configures the shell to automatically activate the base environment.
SHELL ["/bin/bash", "-l", "-c"]
# RUN micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX && \
#     echo "conda activate $ENV_NAME" >> ~/.bashrc
RUN micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX && \
    echo "micromamba activate $ENV_NAME" >> ~/.bashrc

# Set the default environment variable for tools that need it
ENV CONDA_DEFAULT_ENV=$ENV_NAME
ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the application source code
COPY . /app

# Install the s1swotcolocs library
# Ensure your setup.py or pyproject.toml is configured correctly for installation
RUN pip install .

# The environment is now active, so you can run python directly
CMD ["python"]
