# ---- Stage 1: Micromamba Base ----
# This stage just installs micromamba for reuse.
FROM ubuntu:22.04 as micromamba-base

ARG MAMBA_VERSION=1.5.6
ENV MAMBA_ROOT_PREFIX=/opt/conda

RUN apt-get update && apt-get install -y curl bzip2 ca-certificates && \
    curl -L https://micromamba.snakepit.net/api/micromamba/linux-64/${MAMBA_VERSION} | \
    tar -xvj --strip-components=1 -C /usr/local/bin/ bin/micromamba && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ---- Stage 2: The Builder ----
# This stage will use the secret to build the complete conda environment.
# This stage will be discarded and is not part of the final image.
FROM micromamba-base as builder

# --- THIS IS THE CRUCIAL STEP ---
# Declare the build argument that will receive the secret.
ARG GITLAB_CREDS

# Install system dependencies needed for building packages (like git)
RUN apt-get update && apt-get install -y git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy your environment file
COPY environment.yml /tmp/environment.yml

# Create the base environment from your file
RUN micromamba create -y -n myenv -f /tmp/environment.yml && \
    micromamba clean -a -y

# Now, activate the environment and install the private package using the secret.
# The `micromamba run` command executes the command within the specified environment.
# This is where the secret is actually used.
RUN micromamba run -n myenv pip install git+https://${GITLAB_CREDS}@gitlab.ifremer.fr/lops-wave/s1ifr.git


# ---- Stage 3: The Final Image ----
# This is the image you will actually use. It is clean and never saw the secret.
FROM ubuntu:22.04

# Set up environment variables for the final image
ENV MAMBA_ROOT_PREFIX=/opt/conda \
    PATH=/opt/conda/bin:$PATH \
    ENV_NAME=myenv

# Install only the RUNTIME system dependencies. No need for git, curl, etc.
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libxext6 libsm6 libxrender1 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- THE MAGIC STEP ---
# Copy the entire conda environment, with s1ifr already installed, from the builder stage.
COPY --from=builder /opt/conda /opt/conda

# Set up the shell to use the activated environment by default
ENV PATH=$MAMBA_ROOT_PREFIX/envs/$ENV_NAME/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy the application source code
COPY . /app

# Install your local application code into the environment
# The environment already contains s1ifr and all its dependencies
RUN pip install .

# Set the default command for the container
CMD ["python"]
