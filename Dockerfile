FROM python:3.9-slim as compiler
ENV PYTHONUNBUFFERED 1

WORKDIR /app/

RUN python -m venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"

COPY ./requirements.txt /app/requirements.txt
RUN pip install -Ur requirements.txt

FROM python:3.9-slim as runner
WORKDIR /app/
COPY --from=compiler /opt/venv /opt/venv
# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
COPY . /app/
EXPOSE 80
CMD python -m uvicorn main:app --host 0.0.0.0 --port 80 --reload
