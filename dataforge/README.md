# Dataforge

CLI tool to create a synthetic dataset with LLMs.

This tool uses octoai.cloud as LLM as a service.

## Usage

To use `dataforge` directly with `poetry`, you have to do once:

```sh
poetry install
```

The tool `dataforge` needs to call octoai.cloud. You have to create a `.env` file with the [API token](https://octo.ai/docs/getting-started/how-to-create-an-octoai-access-token).

```
OCTOAI_TOKEN=your_token
```

First you can check the number of calls and the cost to run it with the `--dry-run`/`-n` option:

```sh
poetry run dataforge -n --lang=en --chunk-size=2000 --chunk-overlap=100 data/IPCC_AR6_SYR_FullVolume.txt
```

When you are ready, you can run the effective calls:

```sh
poetry run dataforge --lang=en --chunk-size=2000 --chunk-overlap=100 data/IPCC_AR6_SYR_FullVolume.txt
```
