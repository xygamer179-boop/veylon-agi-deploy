#!/usr/bin/env python3
"""
Veylon MoFA Mega‑Agent Hub
Connects all finetuned experts and routes queries to the most relevant ones,
synthesising a unified response – just like the built‑in 'mega' mode, but
using your external finetuneData experts.
"""

import sys
import os
import json
import re
from typing import List, Dict, Optional

# Import Veylon core (make sure veylon_agi_v4.py is in the same directory)
try:
    import veylon_agi_v4 as veylon
except ImportError:
    print("[ERROR] veylon_agi_v4.py not found. Place this file in the same folder.")
    sys.exit(1)

# ── Expert Router ──────────────────────────────────────────────────────
class AgentHub:
    """
    Routes queries to the best finetuned experts and combines their outputs.
    """

    def __init__(self):
        # Ensure model is loaded
        if veylon._model is None:
            veylon._load_model()
        self.model = veylon._model
        self.vectorizer = veylon._vectorizer
        self.config = veylon.config
        self.intent_labels = veylon.INTENT_LABELS

        # Expert mapping – same as EXPERT_DOMAINS in the main script
        self.expert_domains = {
            0: ("code",       ["code"]),
            1: ("math",       ["math", "optimize"]),
            2: ("search",     ["search", "rag"]),
            3: ("reasoning",  ["cot", "tot", "brainstorm"]),
            4: ("chat",       ["chat", "greet", "who"]),
            5: ("vision",     ["vision", "imagegen"]),
            6: ("meta",       ["learn", "predict", "config", "explearn", "critique"]),
            7: ("general",    ["react", "grammar"]),
        }

    def encode(self, text: str):
        return veylon._encode(text)

    def route(self, query: str, top_k: int = 2) -> List[int]:
        """
        Use the MoFA gate to find the top‑k most relevant experts for a query.
        Returns a list of expert indices.
        """
        sparse = self.encode(query)
        # Run the GPT backbone and get gate pre‑softmax
        dense = tf.constant(sparse, dtype=tf.float32)
        if len(dense.shape) == 1:
            dense = tf.expand_dims(dense, 0)
        h = self.model.input_proj(dense)
        B = tf.shape(h)[0]
        h = tf.reshape(h, (B, self.model.n_seq, self.model.embed_dim))
        h = h + tf.cast(self.model.pos_emb, h.dtype)
        for blk in self.model.gpt_blocks:
            h = blk(h, training=False)
        h = self.model.final_ln(h, training=False)
        pooled = (h[:, 0, :] + tf.reduce_mean(h, axis=1)) / 2.0
        gates = tf.nn.softmax(tf.matmul(pooled, self.model.W_gate) + self.model.b_gate)
        gates = gates.numpy().flatten()
        top_experts = list(np.argsort(gates)[::-1][:top_k])
        return top_experts

    def query(self, text: str, mode: str = "mega") -> str:
        """
        Main entry point: route to experts and return a unified answer.
        """
        # First, get the intent from the main model (for context)
        sparse = self.encode(text)
        intent, arg = veylon.detect_intent(text, self.model, sparse, self.config.get("temperature", 0.7))
        domain = veylon.detect_domain(text)

        # Get top experts for this query
        experts = self.route(text, top_k=2)

        # Build response sections
        response_parts = []
        response_parts.append(f"### Mega‑Agent Response (MoFA)")
        response_parts.append(f"**Intent:** {intent} | **Domain:** {domain}")
        response_parts.append(f"**Activated Experts:** {', '.join(f'E{e}' for e in experts)}")
        response_parts.append("")

        # Ask each expert to respond using its specialised knowledge
        for eid in experts:
            domain_name, domain_intents = self.expert_domains.get(eid, ("unknown", []))
            response_parts.append(f"**Expert {eid} ({domain_name}):**")
            # Use the built‑in agent that corresponds to the expert's domain
            expert_response = self._call_expert_agent(eid, text, arg, domain, intent)
            response_parts.append(expert_response)
            response_parts.append("")

        # Synthesise with Swarm of Agents (if available)
        try:
            swarm = veylon._soa.deliberate(text, domain, rag_context="", n_rounds=1)
            response_parts.append(f"**Swarm Synthesis:**")
            response_parts.append(swarm)
        except Exception:
            pass

        # Run through Constitutional AI for safety
        full_raw = "\n\n".join(response_parts)
        try:
            reviewed, _ = veylon._cai.apply(text, full_raw, veylon.config["cai_max_revisions"])
        except Exception:
            reviewed = full_raw

        return reviewed

    def _call_expert_agent(self, expert_id: int, query: str, arg: str,
                           domain: str, intent: str) -> str:
        """
        Dispatch to the appropriate sub‑agent based on the expert's domain.
        This mimics the mega‑mode dispatch logic.
        """
        domain_name, _ = self.expert_domains.get(expert_id, ("general", []))
        if domain_name in ("code",):
            code_response, code_reasoning = veylon.coder_agent(query)
            return f"**Coder:**\n{code_reasoning}\n\n{code_response}"
        elif domain_name in ("math",):
            math_response, math_reasoning = veylon.math_agent(query)
            return f"**Math:**\n{math_reasoning}\n\nResult: {math_response}"
        elif domain_name in ("search",):
            results = veylon._fetcher.fetch(query, max_results=3)
            if results:
                return f"**Search:**\n{results[0]['text'][:500]}"
            return "**Search:** No results found."
        elif domain_name in ("reasoning",):
            cot_result = veylon._thinker.think(query, domain,
                                               depth=veylon.config.get("thinking_depth", 6),
                                               verify=True)
            return f"**CoT:**\n{cot_result['answer']}"
        elif domain_name in ("chat",):
            swarm_resp = veylon._soa.deliberate(query, domain, rag_context="", n_rounds=1)
            return f"**Chat:**\n{swarm_resp}"
        elif domain_name in ("vision", "imagegen"):
            # If it's imagegen, try to generate; otherwise do analysis
            if "generate" in query.lower() or "draw" in query.lower() or "paint" in query.lower():
                result = veylon._cppn.generate(query, 
                                               veylon.config.get("cppn_width", 128),
                                               veylon.config.get("cppn_height", 128))
                analysis = ""
                if result["success"] and veylon.HAS_PIL:
                    vis = veylon._vision.analyze_from_path(result["path"])
                    analysis = veylon._vision.format_analysis(vis)
                return veylon._cppn.format_result(result, analysis)
            else:
                vis_result = veylon._vision.format_analysis({"error": "Provide an image URL or path."})
                return f"**Vision:** {vis_result}"
        elif domain_name in ("meta",):
            # Meta covers learn/predict/config etc.
            return f"**Meta:** Handling intent '{intent}' with meta‑agent."
        else:
            # General fallback
            return f"**General:** Processing '{query[:60]}' with general agent."

# ── Quick CLI interface ─────────────────────────────────────────────────
if __name__ == "__main__":
    hub = AgentHub()
    print("Veylon MoFA Mega‑Agent ready. Type a query or 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou > ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            break
        response = hub.query(user_input)
        print("\n" + response)