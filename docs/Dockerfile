FROM squidfunk/mkdocs-material:latest

# Install plugins.
COPY docs/requirements.txt /tmp/
RUN apk add gcc python3-dev musl-dev linux-headers
RUN pip install --disable-pip-version-check --no-cache-dir -r /tmp/requirements.txt
