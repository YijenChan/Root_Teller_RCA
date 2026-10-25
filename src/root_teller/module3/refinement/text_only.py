from __future__ import annotations

import copy
import json


class TextOnlyPolicy:
    def __init__(self, initial, client, budget=20):
        self.roster = list(initial.base_scores)
        self.order = sorted(self.roster, key=lambda entity: -initial.base_scores[entity])
        self.client = client
        self.budget = budget
        self.interactions = 0
        self.closed = False
        self.conversation = [{"role": "user", "content": json.dumps({
            "evidence": initial.raw["evidence"],
            "windows": initial.raw["windows"],
            "initial_ranking": self.order,
        }, sort_keys=True)}]

    def leader(self):
        return None if self.closed or self.interactions >= self.budget else self.order[0]

    def submit(self, feedback_id, inspected_entity, verdict):
        if self.leader() != inspected_entity or verdict not in {"challenge", "final_accept"}:
            raise ValueError("Feedback must target the current candidate")
        self.interactions += 1
        if verdict == "final_accept":
            self.closed = True
            return {"status": "OPERATOR_ACCEPTED", "interactions": self.interactions}
        self.conversation.append({"role": "user", "content":
            f"The operator rejected {inspected_entity} as the root cause. Reconsider the diagnosis using the retained evidence."})
        if self.client is None:
            raise ValueError("Text-only feedback requires a configured model client")

        def validate(value):
            order = value.get("ranked_entities")
            if not isinstance(order, list) or len(order) != len(set(order)) or set(order) != set(self.roster):
                raise ValueError("Text-only synthesis must preserve the candidate roster")
            return {"ranked_entities": order}

        response, metadata = self.client.complete(
            role="text_only_feedback", prompt_version="text-only-feedback-v1",
            system_prompt="Revise the service ranking from the evidence and conversational operator verdicts. Operator verdicts can be wrong. Return JSON with ranked_entities containing every candidate once.",
            payload={"conversation": copy.deepcopy(self.conversation)}, validator=validate,
        )
        self.order = validate(response)["ranked_entities"]
        self.conversation.append({"role": "assistant", "content": json.dumps(response)})
        return {"status": "CONTINUE", "ranking": self.order.copy(),
                "interactions": self.interactions, "client_metadata": metadata}
