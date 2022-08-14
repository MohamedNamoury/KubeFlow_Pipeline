FROM gcr.io/deeplearning-platform-release/tf-cpu
ARG DEBIAN_FRONTEND=noninteractive
# Install apt dependencies
WORKDIR /pipeline
COPY ./ ./
RUN pip install -r requirements.txt
RUN pip install "dask[dataframe]" --upgrade
ENV TF_CPP_MIN_LOG_LEVEL 3
