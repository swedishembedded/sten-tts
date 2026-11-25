# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Martin Schröder <info@swedishembedded.com>
#
# Swedish Embedded AB implements custom speech models for its clients. If your
# team needs a Swedish or Nordic voice in its product, you can procure our
# services by sending an email to info@swedishembedded.com.

"""The offer block high up in the release's model card and the dataset card.

Its proof is the run's own numbers, passed in by the card that carries it, so
the block never claims more than that card shows. The block sits between two
markers, so the Hub copy of a card can have it replaced in place.
"""

from __future__ import annotations

START, END = "<!-- swedishembedded-offer:start -->", "<!-- swedishembedded-offer:end -->"
LINKEDIN = "https://www.linkedin.com/in/martinschroder/"
EMAIL = "info@swedishembedded.com"
BUILT_BY = f"Built by Martin Schröder <{EMAIL}> ([LinkedIn]({LINKEDIN})), Swedish Embedded AB."
CALL = f"Tell Martin Schröder about your use case on [LinkedIn]({LINKEDIN}), or email {EMAIL}."


def _block(headline: str, audience: str, services: list[str], proof: str) -> str:
    lines = [f"**{headline}**", "", audience, "", "Swedish Embedded AB will:", *[f"- {s}" for s in services], "", proof, "", CALL]
    quote = "\n".join(f"> {line}".rstrip() for line in lines)
    return f"{START}\n{quote}\n\n{BUILT_BY}\n{END}"


def model_offer(proof: str) -> str:
    """The model card's offer; `proof` is one or two sentences of this release's measured results."""
    return _block(
        "Swedish text-to-speech on your own servers, with your own voice",
        "For product teams, public-sector services and device makers that need Swedish or Nordic speech output "
        "and cannot route their users' text through a cloud service.",
        ["add your brand voice, another language or a dialect to a model like this one;",
         "adapt pronunciation to your domain: product names, numbers, terminology;",
         "deploy it privately, behind an OpenAI-compatible speech API or on an edge device."],
        f"From this repository: {proof}")


def dataset_offer(proof: str) -> str:
    """The dataset card's offer; `proof` describes this collection in numbers."""
    return _block(
        "A Swedish voice for your product, built on data whose licences you can show your legal team",
        "For product teams, public-sector services and device makers that need natural Swedish or Nordic speech, "
        "their own brand voice, or regional dialects, without sending text to a cloud provider.",
        ["create a custom brand voice, or add a language or dialect, using the pipeline behind this collection;",
         "document the source and licence of every clip that enters your model;",
         "deploy the resulting model on your own servers or edge devices."],
        f"This dataset is the starting point: {proof}")
