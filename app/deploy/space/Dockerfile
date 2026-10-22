FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/* \
 && useradd -m -u 1000 user && mkdir /app && chown user /app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
USER user
COPY --chown=user auscult-trust ./auscult-trust
COPY --chown=user repos ./repos
COPY --chown=user entrypoint.sh .
WORKDIR /app/auscult-trust
ENV PYTHONPATH=. PYTHONUNBUFFERED=1 AT_APP_BUNDLE=app/bundle
EXPOSE 7860
CMD ["bash", "/app/entrypoint.sh"]
