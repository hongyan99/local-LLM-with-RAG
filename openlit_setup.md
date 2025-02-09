# OpenLIT Quick Setup

OpenLIT GitHub [Repo](https://github.com/openlit/openlit)

## Install Docker Desktop:

Follow the instruction at: https://docs.docker.com/desktop/setup/install/windows-install/

## Install OpenLIT Server:

* Create virtual environment
* Get git clone URL from openlit: https://github.com/openlit/openlit
    git clone https://github.com/openlit/openlit.git
* Run docker compose
    cd openlit
    docker compose up -d

## Install OpenLIT and ollama
    pip install openlit
    pip install ollama

## Login to the Dashboard of OpenLIT Using the Default Username
    Username = user@openlit.io
    Password = openlituser

## Register With OpenLIT in Application

Note that the instruction in openlit site suggested to use https://127.0.0.1:4318 but if you have not yet
setup https for your docker instance, you cannot use the https. Just use http instead.

## View Logs

    localhost:3000