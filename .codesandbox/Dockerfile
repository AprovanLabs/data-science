FROM python:3.10.11-slim-bullseye

ENV POETRY_VERSION 1.5.0

RUN set -ex; pip install --no-cache-dir poetry==$POETRY_VERSION;
RUN apt-get update && apt-get install --no-install-recommends -y make

CMD [ "poetry" ]
