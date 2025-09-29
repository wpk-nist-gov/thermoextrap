#/usr/bin/env sh

# NOTE: This will create monkeytype file in $NOTEBOOK_DIR
# it is copied over to $ROOT when done.
# So, run this first, and follow with
#
# > pytest --monkeytype-output=./monkeytype.sqlite3 -x

set -euxo pipefail

ROOT=$(PWD)
NOTEBOOK_DIR=examples/usage/basic
cd $NOTEBOOK_DIR

# convert to script
uv run jupyter nbconvert --to script *.ipynb

# replace get ipython
sd -F "get_ipython().run_line_magic('matplotlib', 'inline')" "" *.py

# remove plt.show
sd -F "plt.show" "" *.py

# run all files
for file in *.py; do
    uv run monkeytype run $file
done

# remove created files
rm *.py

mv -i monkeytype.sqlite3 $ROOT

cd $ROOT
uv run pytest --monkeytype-output=./monkeytype.sqlite3
