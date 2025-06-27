# cmomy.compile for .venv
compile-venv:
    uv run --no-dev -m cmomy.compile

# cmomy.compile for .nox
compile-nox:
    just nox -s test test-notebook -- ++test-no-pytest ++test-run python -m cmomy.compile

# run cmomy.compile for .venv and .nox
compile-all: compile-venv compile-nox
