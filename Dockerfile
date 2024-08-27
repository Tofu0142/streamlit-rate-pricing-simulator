FROM python:3.11-bullseye as builder

WORKDIR /app

COPY . /app

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate

RUN apt-get update --fix-missing \
  && apt-get install gcc -y \
  && apt-get clean

RUN pip install .


FROM python:3.11-slim as deploy
WORKDIR /app

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv && \
  . /opt/venv/bin/activate

RUN groupadd -r app -g1000 &&\
  useradd -r  -u 1000 -g app -d /app -s /sbin/nologin -c "Docker image user" app && \
  chown -R app:app /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/source /app/source
COPY --from=builder /app/*.py /app/
COPY --from=builder /app/requirements.txt /app/

ENV PYTHONPATH="/app/source:$PYTHONPATH"

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]


FROM deploy as test-dependencies
COPY --from=builder /app /app
RUN pip install --no-cache-dir pytest flake8 moto[s3] freezegun coverage surrogate && \
  mkdir /output


FROM test-dependencies as lint
RUN python -m flake8 app tests | tee /output/linting.txt


FROM test-dependencies as test
RUN python -m pytest tests | tee /output/tests.txt
