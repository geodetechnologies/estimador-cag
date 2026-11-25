"""Create one context-rich chunk for each budget component."""

import tiktoken

from .schemas import Budget, Chunk


class JSONStructuralChunker:
    def chunk(self, budgets: list[Budget]) -> list[Chunk]:
        encoding = tiktoken.encoding_for_model("text-embedding-3-small")
        chunks = []
        for budget in budgets:
            for component in budget.components:
                text = (
                    f"[Project: {budget.project_summary}]\n"
                    f"[Client sector: {budget.client_metadata.sector} | Year: {budget.year}"
                    f" | Main tech: {budget.main_technology}]\n\n"
                    f"Component: {component.name}\n"
                    f"Description: {component.description}\n"
                    f"Tech stack: {', '.join(component.tech_stack)}\n"
                    f"Complexity: {component.complexity}\n"
                    f"Estimated hours: {component.estimated_hours}"
                )
                chunks.append(Chunk(
                    chunk_id=f"{budget.budget_id}::{component.component_id}",
                    text=text,
                    metadata={
                        "budget_id": budget.budget_id,
                        "component_id": component.component_id,
                        "client_sector": budget.client_metadata.sector,
                        "main_technology": budget.main_technology,
                        "year": budget.year,
                        "complexity": component.complexity,
                        "estimated_hours": component.estimated_hours,
                    },
                    token_count=len(encoding.encode(text, disallowed_special=())),
                ))
        return chunks
