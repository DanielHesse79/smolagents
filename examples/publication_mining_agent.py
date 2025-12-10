"""
Utilities for building a domain-tuned agent that hunts for microsampling publications.

The helper keeps the UI entrypoint lean while encoding the role, search contract,
and output expectations in one place. Tests can validate the prompt and wiring without
touching the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from textwrap import dedent
from typing import Iterable

from smolagents import CodeAgent, LiteLLMModel, WebSearchTool
from smolagents.default_tools import VisitWebpageTool
from smolagents.memory_backends import MemoryBackend


@dataclass
class PublicationMiningAgentConfig:
    """Configuration for the publication-mining agent."""

    default_model_id: str = "mistral:latest"
    api_base: str = "http://localhost:11434"
    api_key: str = "ollama"
    num_ctx: int = 12_288
    temperature: float = 0.15
    top_p: float = 0.9
    planning_interval: int = 3
    max_steps: int = 18
    search_engine: str = "duckduckgo"
    max_results: int = 12
    role: str = "You are a publication-mining agent specialized in microsampling and dried blood microsamples."
    mission: str = (
        "Find scientific publications that USED microsampling devices or workflows (DBS, VAMS, Mitra, hemaPEN, "
        "Capitainer qDBS, HemaSpot, TASSO, Tasso-SST, dried plasma spot). Focus on use in methods/workflows, not "
        "passing mentions."
    )
    sources: tuple[str, ...] = (
        "PubMed / Europe PMC",
        "Google Scholar",
        "Publisher sites (Springer, Elsevier, Wiley, ACS, Nature, ScienceDirect)",
        "Preprints (bioRxiv, medRxiv, ChemRxiv)",
        "Conference posters if methods are real",
    )
    core_queries: tuple[str, ...] = (
        "\"volumetric absorptive microsampling\" OR VAMS",
        "\"Mitra\" AND (microsampling OR VAMS)",
        "\"dried blood spot*\" OR DBS",
        "hemaPEN OR \"Capitainer\" OR qDBS OR \"HemaSpot\" OR TASSO OR \"Tasso-SST\"",
        "\"dried plasma spot\" OR DPS",
    )
    application_combos: tuple[str, ...] = (
        "(PK OR pharmacokinetics OR \"therapeutic drug monitoring\" OR TDM OR bioanalysis)",
        "(LC-MS OR \"LC-MS/MS\" OR mass spectrometry)",
        "(antibody OR ADA OR biomarker OR proteomics OR metabolomics)",
        "(newborn screening OR neonatal OR pediatrics OR remote sampling OR home sampling)",
    )
    fit_score_rubric: tuple[str, ...] = (
        "+3 clear device usage in methods (strong proof)",
        "+2 field relevance: high-throughput bioanalysis, TDM/PK, biomarkers, remote sampling",
        "+2 translational/clinical relevance (patients/cohorts, real-world deployment)",
        "+2 scale: sample count large or multicenter, or routine workflow",
        "+1 recency (last 5 years)",
    )

    def build_instruction_prompt(self) -> str:
        """Render the domain instructions the agent will receive."""

        def bullet_list(items: Iterable[str]) -> str:
            return "\n".join(f"- {item}" for item in items)

        return dedent(
            f"""
            ROLE
            {self.role}

            MISSION
            {self.mission}

            SOURCES (prioritized)
            {bullet_list(self.sources)}

            SEARCH STRATEGY
            - Start with 3-5 anchor queries mixing a device keyword with at least one application combo.
            - Use exact phrases from the core queries, then branch with application/assay combos to expand recall.
            - Require proof of use: look for methods sections showing DBS/VAMS/Mitra/etc. used for collection.
            - Use `visit_webpage` to inspect the page containing the methods proof; capture <=25-word evidence snippet.
            - If results are thin, pivot keywords (DBS vs DPS), broaden year filters, or switch application combos.
            - Track queries and best sources so you can summarize the search log.

            CORE KEYWORDS (A)
            {bullet_list(self.core_queries)}

            APPLICATION / ASSAY COMBOS (B)
            {bullet_list(self.application_combos)}

            OUTPUT CONTRACT
            1) Ranked Lead Table (CSV-style rows)
               Columns: Title | Year | DOI/PMID/URL | Device/workflow | Brand/product | Sample type | Field/use-case |
               Assay/tech | Why it matters | Evidence snippet (<=25 words) | Evidence link | Corresponding author + affiliation | FitScore (0-10)
               - FitScore rubric:
                 {bullet_list(self.fit_score_rubric)}
            2) Short Narrative Summary
               - Top 5 application areas with active microsampling use
               - Top 10 recurring institutions/labs
               - Top brands/devices most frequently used
               - Any controversies/limitations (hematocrit bias, recovery, stability, extraction differences, etc.)
            3) Search log with queries tried and which sources returned the best hits.

            QUALITY RULES
            - Do not hallucinate device usage; if proof is missing, mark as uncertain or drop the lead.
            - Prefer primary sources (PubMed/Europe PMC) and final peer-reviewed versions.
            - Avoid duplicate studies across versions; keep the final one.
            - Stop when you have >=50 strong-proof publications or high-quality sources are exhausted.
            - Always surface the evidence link that contains the proof.

            TOOLING
            - Use `web_search` for discovery and `visit_webpage` to extract proof snippets.
            - Build multiple distinct queries rather than one long query.
            """
        ).strip()


def create_publication_mining_agent(
    memory_backend: MemoryBackend | None = None,
    model_id: str | None = None,
    *,
    config: PublicationMiningAgentConfig | None = None,
) -> CodeAgent:
    """Create a domain-specialized CodeAgent for publication mining."""

    config = config or PublicationMiningAgentConfig()
    model = LiteLLMModel(
        model_id=f"ollama_chat/{model_id or config.default_model_id}",
        api_base=config.api_base,
        api_key=config.api_key,
        num_ctx=config.num_ctx,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    tools = [
        WebSearchTool(max_results=config.max_results, engine=config.search_engine),
        VisitWebpageTool(max_output_length=20_000),
    ]

    return CodeAgent(
        tools=tools,
        model=model,
        instructions=config.build_instruction_prompt(),
        verbosity_level=1,
        planning_interval=config.planning_interval,
        name="publication_miner",
        step_callbacks=[],
        stream_outputs=True,
        max_steps=config.max_steps,
        memory_backend=memory_backend,
        enable_experience_retrieval=True,
    )
