#!/usr/bin/env bash

source ./.venv/bin/activate

printf "\n\n ===== Running type checker ===== \n\n"
mypy src/

printf "\n\n ===== Running style formatter ===== \n\n"
black src/

printf "\n\n ===== Running tests  ===== \n\n"
pytest src/
