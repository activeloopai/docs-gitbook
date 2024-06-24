---
description: >-
  Guidelines for open source enthusiasts to contribute to our open-source data
  format.
---

# How to Contribute

## How to Contribute to Activeloop Open-Source

#### Deep Lake relies on feedback and contributions from our wonderful community. Let's make it amazing with your help! Any and all contributions are appreciated, including code profiling, refactoring, and tests.

### Providing Feedback

We love feedback! Please [join our Slack Community ](http://slack.activeloop.ai/)or [raise an issue in Github](https://github.com/activeloopai/Hub/issues).

### Getting Started With Development

Clone the repository:

```bash
git clone https://github.com/activeloopai/deeplake 
cd deeplake
```

&#x20;If you are using Linux, install environment dependencies:

```
apt-get -y update
apt-get -y install git wget build-essential python-setuptools python3-dev libjpeg-dev libpng-dev zlib1g-dev
apt install build-essential
```

If you are planning to work on videos, install codecs:

```
apt-get install -y ffmpeg libavcodec-dev libavformat-dev libswscale-dev
```

Install the package locally with plugins and development dependencies:

```
pip install -r deeplake/requirements/plugins.txt
pip install -r deeplake/requirements/tests.txt
pip install -e .
```

Run local tests to ensure everything is correct:

```
pytest -x --local . 
```

#### **Using Docker** (optional)

You can use docker-compose for running tests

```
docker-compose -f ./bin/docker-compose.yaml up --build local
```

and even work inside the docker by building the image and bashing into.

```
docker build -t activeloop-deeplake:latest -f ./bin/Dockerfile.dev .
docker run -it -v $(pwd):/app activeloop-deeplake:latest bash
$ python3 -c "import deeplake"
```

Now changes done on your local files will be directly reflected into the package running inside the docker.&#x20;

### Contributing Guidelines

#### Linting

Deep Lake uses the [black](https://pypi.org/project/black/) python linter. You can auto-format your code by running `pip install black`, and the run `black .` inside the directory you want to format.

#### Docstrings

Deep Lake uses Google Docstrings. Please refer to [this example](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example\_google.html) to learn more.

#### Typing

Deep Lake uses static typing for function arguments/variables for better code readability. Deep Lake has a GitHub action that runs `mypy .`, which runs similar to `pytest .` to check for valid static typing. You can refer to [mypy documentation](https://mypy.readthedocs.io/en/stable/) for more information.

#### Testing

Deep Lake uses [pytest](https://docs.pytest.org/en/6.2.x/) for tests. In order to make it easier to contribute, Deep Lake also has a set of custom options defined [here](https://github.com/activeloopai/Hub/tree/main/hub/tests).

#### Prerequisites

* Understand how to write [pytest](https://docs.pytest.org/en/6.2.x/) tests.
* Understand what a [pytest fixture](https://docs.pytest.org/en/6.2.x/fixture.html) is.
* Understand what [pytest parametrizations](https://docs.pytest.org/en/6.2.x/parametrize.html) are.

#### Options

To see a list of Deep Lake's custom pytest options, run this command: `pytest -h | sed -En '/custom options:/,/\[pytest\] ini\-options/p'`.

#### Fixtures

You can find more information on pytest fixtures [here](https://docs.pytest.org/en/6.2.x/fixture.html).

* `memory_storage`: If `--memory-skip` is provided, tests with this fixture will be skipped. Otherwise, the test will run with only a `MemoryProvider`.
* `local_storage`: If `--local` is **not** provided, tests with this fixture will be skipped. Otherwise, the test will run with only a `LocalProvider`.
* `s3_storage`: If `--s3` is **not** provided, tests with this fixture will be skipped. Otherwise, the test will run with only an `S3Provider`.
* `storage`: All tests that use the `storage` fixture will be parametrized with the enabled `StorageProvider`s (enabled via options defined below). If `--cache-chains` is provided, `storage` may also be a cache chain. Cache chains have the same interface as `StorageProvider`, but instead of just a single provider, it is multiple chained in a sequence, where the last provider in the chain is considered the actual storage.
* `ds`: The same as the `storage` fixture, but the storages that are parametrized are wrapped with a `Dataset`.

Each `StorageProvider`/`Dataset` that is created for a test via a fixture will automatically have a root created, and it will be destroyed after the test. If you want to keep this data after the test run, you can use the `--keep-storage` option.

#### **Fixture Examples**

Single storage provider fixture:

```python
def test_memory(memory_storage):
    # Test will skip if `--memory-skip` is provided
    memory_storage["key"] = b"1234"  # This data will only be stored in memory

def test_local(local_storage):
    # Test will skip if `--local` is not provided
    memory_storage["key"] = b"1234"  # This data will only be stored locally

def test_local(s3_storage):
    # Test will skip if `--s3` is not provided
    # Test will fail if credentials are not provided
    memory_storage["key"] = b"1234"  # This data will only be stored in s3
```

Multiple storage providers/cache chains:

```python
from deeplake.core.tests.common import parametrize_all_storages, parametrize_all_caches, parametrize_all_storages_and_caches

@parametrize_all_storages
def test_storage(storage):
    # Storage will be parametrized with all enabled `StorageProvider`s
    pass

@parametrize_all_caches
def test_caches(storage):
    # Storage will be parametrized with all common caches containing enabled `StorageProvider`s
    pass

@parametrize_all_storages_and_caches
def test_storages_and_caches(storage):
    # Storage will be parametrized with all enabled `StorageProvider`s and common caches containing enabled `StorageProvider`s
    pass
```

Dataset storage providers/cache chains:

```
from deeplake.core.tests.common import parametrize_all_dataset_storages, parametrize_all_dataset_storages_and_caches

@parametrize_all_dataset_storages
def test_dataset(ds):
    # `ds` will be parametrized with 1 `Dataset` object per enabled `StorageProvider`
    pass

@parametrize_all_dataset_storages_and_caches
def test_dataset(ds):
    # `ds` will be parametrized with 1 `Dataset` object per enabled `StorageProvider` and all cache chains containing enabled `StorageProvider`s
    pass
```

### Benchmarks

Deep Lake uses [pytest-benchmark](https://pytest-benchmark.readthedocs.io/en/latest/usage.html) for benchmarking, which is a plugin for [pytest](https://docs.pytest.org/en/6.2.x/).

### Here's a list of people who are building the future of data!

Deep Lake would not be possible without the work of our community.

![Activeloop Deep Lake open-source contributors](<../.gitbook/assets/68747470733a2f2f636f6e747269622e726f636b732f696d6167653f7265706f3d6163746976656c6f6f7061692f687562 (1).svg>)

