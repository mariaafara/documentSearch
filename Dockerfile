# Pull the offical docker image: Base image
FROM python:3.8

# Set the work Directory to /search in the container
WORKDIR /documentSearch

# ADD and install requirements
ADD requirements.txt /documentSearch
RUN pip install -r requirements.txt && pip install -U spacy[lookups] && python -m spacy download en_core_web_sm

ADD search /documentSearch/search

ADD main.py /documentSearch

ADD api /documentSearch/api

ENV PYTHONPATH=/documentSearch

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0"]

