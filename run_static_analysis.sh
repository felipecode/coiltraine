#!/usr/bin/env bash

#run pylint
python3 -m pylint **/*.py $@ || exit 0

#run flake8
python3 -m flake8 **/*.py $@ || exit 0
