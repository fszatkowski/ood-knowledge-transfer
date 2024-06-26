#!/bin/bash

isort --profile black src
black src
flake8 src --max-line-length 120
