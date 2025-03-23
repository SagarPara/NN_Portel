#getting OS and Python image from DockerHub
FROM python:3.11-slim-bullseye

WORKDIR /docker

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./ ./

CMD ["python3", "-m", "flask", "--app", "hello_portel_single_value_postman", "run", "--host=0.0.0.0"]