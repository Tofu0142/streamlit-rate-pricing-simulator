# template-streamlit

Streamlit app template


## Get Your Very Own Streamlit Repository

Let's assume we're creating a project name with the name `streamlit-rate-cache-shopping`.

1. `git clone git@github.com:getaroom/terraform-github.git`
2. `git checkout -b add-streamlit-rate-cache-shopping`
3. `cp streamlit_room_offers_repository.tf streamlit_rate_cache_shopping_repository.tf`
4. Brace yourself, we're gonna use some `sed`.  `\1` refers to the matched delimeter `(_|-)`.
```bash
sed -i -E "s/streamlit(_|-)room(_|-)offers/streamlit\1rate\1cache\1shopping/g" streamlit_rate_cache_shopping_repository.tf
```
6. Update any other relevant details such as a descriptions, etc.
7. `git commit -m "add streamlit-rate-cache-shopping repo"`
8. `git push -u`
9. Create a pull request on https://github.com/getaroom/terraform-github


## Pyenv

Assumes `pyenv` is already setup.

Create virtual env for streamlit project.

```bash
echo "3.11.4/envs/streamlit-my-app" > .python-version
pyenv virtualenv streamlit-my-app 3.11.4
```

## Setup

```bash
pip install pip-tools
pip-compile --extra testing -o testing-requirements.txt pyproject.toml
pip-sync testing-requirements.txt
```

## Run Streamlit Locally

```bash
streamlit run main.py
```


## Update Dependencies

```bash
pip-compile --upgrade
pip-sync
```
