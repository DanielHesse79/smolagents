from examples.publication_mining_agent import (
    PublicationMiningAgentConfig,
    create_publication_mining_agent,
)


def test_instruction_prompt_contains_contract_and_strategy():
    config = PublicationMiningAgentConfig()

    prompt = config.build_instruction_prompt()

    assert "publication-mining agent" in prompt
    assert "OUTPUT CONTRACT" in prompt
    assert "Ranked Lead Table" in prompt
    assert "Search log" in prompt
    assert "web_search" in prompt


def test_agent_wiring_matches_config():
    config = PublicationMiningAgentConfig(max_steps=7, search_engine="bing", max_results=5)

    agent = create_publication_mining_agent(config=config)

    assert agent.name == "publication_miner"
    assert agent.max_steps == 7
    assert "microsampling" in agent.instructions
    tool_names = set(agent.tools.keys())
    assert {"web_search", "visit_webpage"}.issubset(tool_names)
    assert agent.model.model_id.endswith(config.default_model_id) or agent.model.model_id.endswith(
        f"/{config.default_model_id}"
    )
