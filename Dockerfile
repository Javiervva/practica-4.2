FROM python:3.11

WORKDIR /myapp

COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY ./app app
COPY ./run.py run.py

EXPOSE 8080
CMD ["python", "run.py", "8080"]