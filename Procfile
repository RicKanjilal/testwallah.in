web: gunicorn app:app \
  --workers=${WEB_CONCURRENCY:-2} \
  --threads=${GUNICORN_THREADS:-8} \
  --timeout=${GUNICORN_TIMEOUT:-300} \
  --keep-alive=5 \
  --preload \
  --bind 0.0.0.0:${PORT:-10000}
