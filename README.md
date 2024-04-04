# Dat'apéro - 21/03/2024, Angers

Ce dépôt contient les fichiers de :
- la présentation _"Fine-tuning d'un LLM"_
    - [Présentation en HTML](./presentation/index.html)
    - [Présentation en PDF](./presentation/dist/presentation.pdf)
- les notebooks du TP
    - [fine-tuning](./notebooks/finetuning_metropole.ipynb)
    - [inférence](./notebooks/inference_metropole.ipynb)

Les modèles générés sont disponibles sur Huggingface :
- [kuhess/hermes-2-pro-mistral-7b-metropole-4bit-gguf](https://huggingface.co/kuhess/hermes-2-pro-mistral-7b-metropole-4bit-gguf)
- [kuhess/hermes-2-pro-mistral-7b-metropole](https://huggingface.co/kuhess/hermes-2-pro-mistral-7b-metropole)

## Sources des données

J'ai utilisé quelques numéros du [journal Métropole](https://www.angersloiremetropole.fr/medias/journal-metropole/index.html) (janvier 2023 -> mars 2024) pour générer un [dataset de quelques milliers de question/réponse](https://github.com/kuhess/datapero-finetuning-llm/blob/main/notebooks/data/metropole.jsonl) au sujet d'Angers Loire Métropole.

J'ai également utilisé le dernier rapport du GIEC (AR6 Full Report) pour tester [`dataforge`](./dataforge/).
