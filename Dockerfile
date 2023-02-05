# Pull the offical docker image: Base image
FROM python:3.8

# Set the work Directory to /search in the container
WORKDIR /documentSearch

# ADD and install requirements
ADD requirements.txt /documentSearch
RUN pip install -r requirements.txt

# ADD data and tests
ADD data /documentSearch/data
ADD tests /documentSearch/tests

# ADD and install package

ADD search /documentSearch/search

ADD main.py /documentSearch

#ADD README.md /documentSearch
#ADD setup.py /documentSearch
#RUN pip install /documentSearch



