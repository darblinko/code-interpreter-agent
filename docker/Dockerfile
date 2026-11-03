# Image for the docker executor with the libraries the examples use.
# Build: docker build -t cia-python docker/
# Use:   cia --executor docker --docker-image cia-python
FROM python:3.12-slim
RUN pip install --no-cache-dir numpy matplotlib
RUN useradd --uid 10001 --create-home runner
