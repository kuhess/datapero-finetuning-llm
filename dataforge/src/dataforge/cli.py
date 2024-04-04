from __future__ import annotations

import asyncio
import json
import os.path
import re
from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import aiometer
import click
from dotenv import load_dotenv
from jsonschema import validate
from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders.text import TextLoader
from loguru import logger
from octoai.chat import ChatCompletion
from octoai.client import Client
from tqdm import tqdm

from dataforge.octoai.chat_async import ChatAsync
from dataforge.prompts.prompts import PromptConfig, generate_messages

if TYPE_CHECKING:
    from langchain_core.documents import Document

OCTOAI_PRICING_INPUT_PER_MILLION_TOKEN = 0.30
OCTOAI_PRICING_OUTPUT_PER_MILLION_TOKEN = 0.50

MODEL = "nous-hermes-2-mixtral-8x7b-dpo"


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


def load_documents(paths: list[Path]) -> Generator[Document, None, None]:
    num_docs = len(paths)
    logger.trace(f"load {num_docs} document{'s' if num_docs != 1 else ''}")
    for path in paths:
        txt_loader = TextLoader(str(path))
        yield txt_loader.load()[0]


async def save_completion(
    result: dict[str, Any],
    run_config: RunConfig,
) -> Path:
    filename = (
        "_".join(
            [
                run_config.id,
                run_config.doc_name,
                f"{run_config.chunk_idx:03d}",
                run_config.prompt_config.name,
            ]
        )
        + ".jsonl"
    )
    output_path = run_config.output_directory / filename

    output_directory = output_path.parent
    if not os.path.exists(output_directory):
        logger.trace(f"create directory {output_directory}")
        os.makedirs(output_directory)

    logger.trace(f"write {len(result)} lines to {output_path}")
    with open(output_path, "w") as output:
        for obj in result:
            output.write(json.dumps(obj))
            output.write("\n")

    return output_path


def parse_json(text: str, json_schema: dict[str, Any]) -> dict[str, Any] | None:
    # clean text
    begin_json = """```json"""
    end_json = "```"
    # remove text before json
    if begin_json in text:
        text = text.split(begin_json)[1]
    # remove text after json
    if end_json in text:
        text = text.split(end_json)[0]
    # remove all backslash
    # text = text.replace("\\", "")
    # keep only text between []
    parts = re.findall(r"(\[.*\])", text, re.DOTALL)
    if len(parts) > 0:
        text = parts[0]
    # translate field if necessary
    text = text.replace('"réponse"', '"answer"')

    try:
        logger.trace(f"going to parse {text} with schema={json_schema}")
        data = json.loads(text)
        validate(data, json_schema)
        return data
    except json.JSONDecodeError as e:
        logger.warning("cannot parse JSON from text:", exception=e)
        return None
    except Exception as e:
        logger.warning("validation error on JSON:", exception=e)
        return None


async def run_completion(
    client: Client,
    run_config: RunConfig,
) -> dict[str, Any] | None:
    chat_async = ChatAsync(client)

    async def query() -> ChatCompletion:
        future_resp = await chat_async.completions.create(
            messages=run_config.messages,
            model=run_config.model,
            max_tokens=run_config.max_tokens,
            presence_penalty=0,
            temperature=0.1,
            top_p=0.9,
        )
        logger.trace(
            f"send request {future_resp.response_id} for "
            f"doc {run_config.doc_name}, "
            f"chunk #{run_config.chunk_idx}, "
            f"prompt {run_config.prompt_config.name}"
        )

        # waiting loop
        while not client.is_future_ready(future_resp):
            await asyncio.sleep(5)

        resp = client.get_future_result(future_resp)
        return ChatCompletion(**resp)

    completion = await query()
    result = parse_json(
        completion.choices[0].message.content, run_config.prompt_config.json_schema
    )
    if result is None:
        # try one more time
        logger.trace(
            "cannot get proper JSON completion on first try, "
            f"going to retry once: {completion.choices[0].message.content}"
        )
        completion = await query()
        result = parse_json(
            completion.choices[0].message.content, run_config.prompt_config.json_schema
        )
        if result is None:
            logger.error(
                f"cannot get a proper JSON completion: {completion.choices[0].message.content}"
            )
            return None

    await save_completion(result, run_config)

    return result


@dataclass
class RunConfig:
    id: str
    model: str
    messages: list[dict[str, str]]
    prompt_config: PromptConfig
    max_tokens: int
    output_directory: Path
    doc_name: str
    chunk_idx: int


def get_doc_name(chunk: Document):
    filename = os.path.basename(chunk.metadata["source"])
    return os.path.splitext(filename)[0]


@click.command()
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(exists=True, file_okay=True, readable=True, path_type=Path),
)
@click.option(
    "--output-dir", default="output", type=click.Path(dir_okay=True, path_type=Path)
)
@click.option("-n", "--dry-run", is_flag=True)
@click.option("--chunk-size", default=4000)
@click.option("--chunk-overlap", default=200)
@click.option("--max-tokens", default=4000)
@click.option("--lang", default="en")
@coro
async def cli(
    input_files: list[click.Path],
    output_dir: Path,
    dry_run: bool,
    chunk_size: int,
    chunk_overlap: int,
    max_tokens: int,
    lang: str,
) -> None:
    load_dotenv()

    docs = load_documents(input_files)
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)

    prompt_configs = PromptConfig.load_all(lang)
    prompt_configs = [
        conf
        for conf in prompt_configs
        if conf.name
        in {
            "simple-quiz",
            "summaries-one",
            "metadata",
            "title-questions",
            "summaries-three",
            "you-are-generating-fine-tuning-data",
        }
    ]

    num_chunks = len(split_docs)
    num_prompts = len(prompt_configs)
    num_calls = num_prompts * num_chunks

    if dry_run:
        input_price = (
            num_calls * chunk_size * OCTOAI_PRICING_INPUT_PER_MILLION_TOKEN / 1e6
        )
        max_output_price = num_calls * OCTOAI_PRICING_OUTPUT_PER_MILLION_TOKEN / 1e6
        click.echo(f"Would send {num_calls} API calls with {chunk_size} tokens each.")
        click.echo(f"Would cost at most ${input_price + max_output_price:.3f}")
        return

    client = Client()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_configs = []
    for i, chunk in enumerate(split_docs):
        for prompt_config in prompt_configs:
            run_configs.append(
                RunConfig(
                    id=run_id,
                    model=MODEL,
                    messages=generate_messages(
                        prompt_config,
                        {"document_chunk": chunk},
                    ),
                    prompt_config=prompt_config,
                    max_tokens=max_tokens,
                    output_directory=output_dir,
                    doc_name=get_doc_name(chunk),
                    chunk_idx=i + 1,
                )
            )

    await aiometer.run_on_each(
        partial(run_completion, client),
        tqdm(run_configs),
        max_per_second=120 / 60,  # 10 per minute
    )
