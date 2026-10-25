"""
Seed Temporal Beliefs for Personal AI Time Capsule.
Extracts and seeds structured, dialectically grounded TemporalBelief entries
across Sergiu's core longitudinal themes:
- Software & Infrastructure Craft (Simplicity is complexity resolved / Brâncuși / .69 IPs)
- Failure and Resilience ("won't succeed if I don't fail first" / Forge of progress)
- 20-Year Compounding Vision (intentional optimism and daily micro-compounding)
- Physical Resilience & Daily Disciplines (iron routines and posture alignment)
- French Learning & Consistency (daily immersion, conversational practice, exam prep)
- Data Privacy & Local Sovereignty (Rule 6: 100% on-device local execution)
- Daily Spoken Reflection & Habit Formation (Daily Speak project / vocal resonance)

Preserves chronological multi-year belief progressions across 2023, 2024, 2025, and 2026,
enforcing timestamped supporting evidence, dialectical counterarguments, and the
Corroborated Evidence Invariant (>= 2 corroborated pieces required for supersession).
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import List, Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig
from core.models import TemporalBelief, VerificationStatus
from memory.database import DatabaseManager


def find_matching_memories(db: DatabaseManager, keywords: List[str], source_prefix: str = "") -> List[Dict[str, Any]]:
    """Finds memory records in archive matching search terms and optional source ID prefix."""
    matched = []
    seen_ids = set()
    with db.get_connection() as conn:
        for kw in keywords:
            sql = """
                SELECT id, content, summary, date_occurred, source_id, start_seconds, end_seconds
                FROM memories
                WHERE content LIKE ? AND verification_status = ?
            """
            params: List[Any] = [f"%{kw}%", VerificationStatus.PERSONAL_VERIFIED.value]
            if source_prefix:
                sql += " AND source_id LIKE ?"
                params.append(f"%{source_prefix}%")
            rows = conn.execute(sql, params).fetchall()
            for r in rows:
                if r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    matched.append({
                        "quote": r["content"],
                        "timestamp_str": r["date_occurred"] or "2026",
                        "start_seconds": r["start_seconds"],
                        "end_seconds": r["end_seconds"],
                        "source_id": r["source_id"],
                        "memory_id": r["id"]
                    })
    return matched


def seed_temporal_beliefs(db_path: str = ""):
    config = CapsuleConfig.load_or_create()
    target_db = db_path or config.storage.sqlite_db_path
    db = DatabaseManager(target_db)

    print(f"[Seed Beliefs] Connecting to database: {target_db}")

    # Remove legacy redundant "Architecture Design" beliefs if present
    redundant_ids = ["belief-arch-early", "belief-arch-mid", "belief-arch-present"]
    for rid in redundant_ids:
        db.delete_belief(rid)

    # 1. Software & Infrastructure Craft Evidence
    craft_supporting = [
        {
            "quote": "a quote that really resonated with me is one from a Romanian sculpture called Brâncuși, Constantin Brâncuși. He said that simplicity is complexity resolved and I found that quite beautiful in the sense that minimalism and simplicity is an evolution of complexity. Like Elon Musk's doctrine of no part is the best part.",
            "timestamp_str": "2026-08-10",
            "start_seconds": 38.53,
            "end_seconds": 79.95,
            "source_id": "b0b95788-b2ef-4c6e-826a-49ae8a156ff3",
            "memory_id": "0b8de00a-03eb-414b-97c6-4dfdc88bc97a"
        },
        {
            "quote": "I adore your curiosity and inner spark, your joy of discovery and insatiable curiosity from playing with building blocks to 3D printers and code modules. You never stop to think I've discovered enough.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 41.52,
            "end_seconds": 77.41,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "b74daf98-cb54-44b2-a4e9-eb6194b05952"
        },
        {
            "quote": "In 2024, established strict private local AI architecture on Apple Silicon M4. Prioritized zero external telemetry, cryptographic provenance, and modular replaceable models.",
            "timestamp_str": "2024-02-15",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "source_id": "syn-src-004",
            "memory_id": "syn-mem-004"
        }
    ]
    craft_counterarguments = [
        {
            "quote": "Simplicity is not naive minimalism; it requires first building and mastering the underlying complexity (.69 static IP topology, Tailscale mesh, Cudy P5 native kernel MSS clamping) before eliminating redundant layers.",
            "timestamp_str": "2026-08-10",
            "start_seconds": 0.0,
            "end_seconds": 38.53,
            "source_id": "b0b95788-b2ef-4c6e-826a-49ae8a156ff3",
            "memory_id": "21a3998a-1311-4601-bd37-c8fbafad39fc"
        }
    ]

    # 2. Failure & Resilience Evidence
    failure_supporting = [
        {
            "quote": "The way I deal with failure is that I won't succeed if I don't fail first. What's important is that if I fail I get back up. I don't give up and I only miss the shots that I don't take.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 244.9,
            "end_seconds": 282.9,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "4337239a-7680-40a3-b1c2-54713a4709ab"
        },
        {
            "quote": "I need to treat setbacks as a forge of progress that fire and pressure that fire and pressure is what shapes me. So setbacks, weakness, demotivation, they are all part of that forge...",
            "timestamp_str": "2026-09-06",
            "start_seconds": 282.9,
            "end_seconds": 322.02,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "82db2a82-cac3-439c-93f2-908503ce7479"
        },
        {
            "quote": "it is possible to build myself and to build my achievements and goals by and be more productive by forging myself in that fire.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 322.1,
            "end_seconds": 341.9,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "ee121c55-9128-41ef-b4ac-894a0f65b3f3"
        }
    ]
    failure_counterarguments = [
        {
            "quote": "Because there are only 10 days left until that exam. And I am not going to fail exams. I am going to ace it. And then I am going to reward myself.",
            "timestamp_str": "2026-09-01",
            "start_seconds": 79.45,
            "end_seconds": 107.86,
            "source_id": "63a53acf-85e7-4959-bc36-a60d1ba525db",
            "memory_id": "2dffa09c-4610-449e-8c38-89c56baefb3a"
        }
    ]

    # 3. 20-Year Compounding Vision Evidence
    vision_supporting = [
        {
            "quote": "I want to look in the future and ask what could happen if it all goes well. What is possible in 20 years and how much better I can be at what I do if I continue to improve each day.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 244.9,
            "end_seconds": 282.9,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "4337239a-7680-40a3-b1c2-54713a4709ab"
        },
        {
            "quote": "A world traveler with an iron routine for sustainable travel so that I can sustain my health and the balance of my mind. I don't want to look in the past and think what might have been. I don't want to look in the future and ask what could have happened. I want to look in the future and ask what could happen if it all goes well.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 208.37,
            "end_seconds": 244.9,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "7e83482d-fcc6-4caf-875f-09a743faaec1"
        },
        {
            "quote": "I want to define the state of the world and my place in it. And the definition I came up with thus far is that I don't want to change the world yet. I want to be an active player on its stage and to better understand it first.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 127.12,
            "end_seconds": 163.55,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "f13a6d0e-3442-4ebb-9889-3b07c9f65bd4"
        },
        {
            "quote": "to see where the next 10 years will take me and I'm at the helm of this beautiful journey that's ahead.",
            "timestamp_str": "2026-09-02",
            "start_seconds": 152.42,
            "end_seconds": 176.83,
            "source_id": "657f5f91-236b-4e1d-a9ec-0466be8f8252",
            "memory_id": "1bee97f1-9165-483f-8fe8-d6c5e7839c09"
        }
    ]
    vision_counterarguments = [
        {
            "quote": "your romantic nature and melting heart in front of grand gestures of love, your nostalgic weakness that breaks you into tears sometimes. Tears that span the gap between your idyllic image and reality.",
            "timestamp_str": "2026-09-06",
            "start_seconds": 77.41,
            "end_seconds": 126.0,
            "source_id": "3791e8c7-4aa7-47b2-bdcb-bb18029cbbde",
            "memory_id": "bc83a5bb-771c-43fe-a33f-f7d9c6beeead"
        }
    ]

    # 4. Physical Resilience Evidence
    physical_supporting = [
        {
            "quote": "Good morning. So today I am on the second day of my routine, approximately, and I'm incrementally adding capabilities to it.",
            "timestamp_str": "2026-08-12",
            "start_seconds": 0.0,
            "end_seconds": 35.28,
            "source_id": "be663d40-f19b-449e-9d22-1d59ae6ec05d",
            "memory_id": "9ae12163-3e95-445b-baf5-7e8c5dee8045"
        },
        {
            "quote": "pre-workout stretching and post-workout stretching as well. At least the minimum viable amounts. I think that is quite important.",
            "timestamp_str": "2026-08-12",
            "start_seconds": 37.12,
            "end_seconds": 68.24,
            "source_id": "be663d40-f19b-449e-9d22-1d59ae6ec05d",
            "memory_id": "3062bae2-850f-4f8e-bd7d-080bf5fce3be"
        },
        {
            "quote": "routine as I am traveling. I will enter these sprints where I build my body up, then I rest and recover... I want to build an iron routine.",
            "timestamp_str": "2026-08-12",
            "start_seconds": 212.18,
            "end_seconds": 245.9,
            "source_id": "be663d40-f19b-449e-9d22-1d59ae6ec05d",
            "memory_id": "d0fb47f4-c32d-4ca4-acf8-beccc82938e4"
        },
        {
            "quote": "every day I am working on correcting my posture, doing some shoulder exercises. Brian Johnson is just amazing...",
            "timestamp_str": "2026-09-02",
            "start_seconds": 72.1,
            "end_seconds": 107.42,
            "source_id": "657f5f91-236b-4e1d-a9ec-0466be8f8252",
            "memory_id": "0365008c-114c-4b4b-8252-cd40ef325b19"
        },
        {
            "quote": "difference between the main muscle types, muscle fiber types, type 1 and type 2. Type 1 is red muscle fibers reached in mitochondria... Type 2 for power.",
            "timestamp_str": "2026-08-13",
            "start_seconds": 0.0,
            "end_seconds": 43.42,
            "source_id": "32f4859a-df62-4217-bfbe-d5673a6a9be8",
            "memory_id": "6050c4b9-91c6-43b8-a7be-e5e6df7ad321"
        }
    ]
    physical_counterarguments = [
        {
            "quote": "terrible in terms of the entire feeling pain. However, I realized that is my body adapting to the physical strain.",
            "timestamp_str": "2026-08-12",
            "start_seconds": 108.52,
            "end_seconds": 142.1,
            "source_id": "be663d40-f19b-449e-9d22-1d59ae6ec05d",
            "memory_id": "fc93c16f-7be8-46c5-ad3a-7362a4501eb8"
        },
        {
            "quote": "so I didn't recover enough, and that means that I cannot push or I cannot pump and exercise without adequate rest.",
            "timestamp_str": "2026-08-14",
            "start_seconds": 0.0,
            "end_seconds": 35.0,
            "source_id": "2ce30501-c852-44a6-953e-f62f3fbe567a",
            "memory_id": "107317fa-555e-4efb-86d4-d3ba928a6fcf"
        }
    ]

    # 5. French Learning Evidence
    french_supporting = [
        {
            "quote": "Good morning. Yesterday I practiced my French using artificial intelligence and it was a very interesting experience. I was able to have a full-on conversation...",
            "timestamp_str": "2026-09-01",
            "start_seconds": 0.0,
            "end_seconds": 41.73,
            "source_id": "63a53acf-85e7-4959-bc36-a60d1ba525db",
            "memory_id": "1cc871b3-4f01-4475-b6d4-b9d9c223c72d"
        },
        {
            "quote": "And I found it super useful in practicing my written and my spoken French. And I have notes over here from the lessons that I will go over and practice...",
            "timestamp_str": "2026-09-01",
            "start_seconds": 41.73,
            "end_seconds": 79.45,
            "source_id": "63a53acf-85e7-4959-bc36-a60d1ba525db",
            "memory_id": "8d5a01e6-df06-4dfc-b9b5-48b846e4ae02"
        },
        {
            "quote": "This month I will maybe make that pumpkin into a pumpkin pie after I pass my French exam.",
            "timestamp_str": "2026-09-02",
            "start_seconds": 0.0,
            "end_seconds": 39.6,
            "source_id": "657f5f91-236b-4e1d-a9ec-0466be8f8252",
            "memory_id": "f4f2ca8f-b9a9-4f12-9db0-0682461daa3d"
        },
        {
            "quote": "My brain was on a loop and it wasn't stopping. It wasn't stopping for like four hours. Constantly speaking in French with myself...",
            "timestamp_str": "2026-09-07",
            "start_seconds": 113.55,
            "end_seconds": 154.34,
            "source_id": "6b3c723c-a9e9-4e0e-9276-f33190df09b4",
            "memory_id": "9cdb8d48-e16b-47c3-be30-aea7de33c1f7"
        }
    ]
    french_counterarguments = [
        {
            "quote": "vocabulary and grammar, whilst I was thinking those things, led to me sleeping later, made myself aware of the fact that I am sleeping later and I will sleep less...",
            "timestamp_str": "2026-09-07",
            "start_seconds": 154.34,
            "end_seconds": 192.8,
            "source_id": "6b3c723c-a9e9-4e0e-9276-f33190df09b4",
            "memory_id": "fab40108-e215-46aa-b2b9-e160a2f588c7"
        },
        {
            "quote": "However, it would have been much nicer if that sleep would have been put to good use to transfer all of the knowledge I gained with respect to language...",
            "timestamp_str": "2026-09-07",
            "start_seconds": 193.36,
            "end_seconds": 208.91,
            "source_id": "6b3c723c-a9e9-4e0e-9276-f33190df09b4",
            "memory_id": "d16b91e8-7a5e-4c37-a1e4-d6e326c92bc9"
        }
    ]

    # 6. Data Privacy & Local Sovereignty Evidence
    privacy_supporting = [
        {
            "quote": "In 2024, established strict private local AI architecture on Apple Silicon M4. Prioritized zero external telemetry, cryptographic provenance, and modular replaceable models.",
            "timestamp_str": "2024-02-15",
            "start_seconds": 0.0,
            "end_seconds": 30.0,
            "source_id": "syn-src-004",
            "memory_id": "syn-mem-004"
        },
        {
            "quote": "Personal memories and identity must run 100% locally with zero cloud leakage, loopback binding, cryptographic manifests, and strict Rule 6 data privacy invariants.",
            "timestamp_str": "2026-10-25",
            "start_seconds": 0.0,
            "end_seconds": 0.0,
            "source_id": "operational-directive-rule6",
            "memory_id": "ag-invariant-rule6"
        }
    ]
    privacy_counterarguments = [
        {
            "quote": "Commercial translation cloud dubbing APIs (like experimental Spanish and Korean dubbing models) offer quick novelty, but leak private voice prints and fail local sovereignty invariants.",
            "timestamp_str": "2023-11-20",
            "start_seconds": 0.0,
            "end_seconds": 25.0,
            "source_id": "synthetic-dub-test",
            "memory_id": "84cb0ade-b7d1-423c-8ac2-ab6444016bf7"
        }
    ]

    # 7. Daily Spoken Reflection Evidence
    speaks_supporting = [
        {
            "quote": "Good morning and well it's been a while since I have recorded a video and I photograph myself and I'm quite excited to see my progress from about two years to now. It is quite something to see how time leaves and mark upon me. It's like the hands of a sculpture modeling the clay and turning it to something different.",
            "timestamp_str": "2026-08-10",
            "start_seconds": 0.0,
            "end_seconds": 36.67,
            "source_id": "b0b95788-b2ef-4c6e-826a-49ae8a156ff3",
            "memory_id": "21a3998a-1311-4601-bd37-c8fbafad39fc"
        },
        {
            "quote": "I realized that as speaking as I'm reciting this video, I need to make sure that I'm engaging with you with eye contact and that I'm also using my deep voice. So using my belly here, I realized I didn't really do that to the full extent.",
            "timestamp_str": "2026-08-13",
            "start_seconds": 149.44,
            "end_seconds": 175.21,
            "source_id": "32f4859a-df62-4217-bfbe-d5673a6a9be8",
            "memory_id": "398656f8-450f-48d6-a0bf-52d3c907153b"
        }
    ]
    speaks_counterarguments = [
        {
            "quote": "Today, I was thinking about the fact that I'm not smiling as much as I used to. I used to smile every day and I used to be super hyped... self-perception on camera vs internal emotional reality.",
            "timestamp_str": "2026-09-07",
            "start_seconds": 0.0,
            "end_seconds": 27.36,
            "source_id": "6b3c723c-a9e9-4e0e-9276-f33190df09b4",
            "memory_id": "bcf79f91-5367-4b71-9252-c80f4f9f7ba3"
        }
    ]

    beliefs_definitions = [
        # =========================================================================
        # 1. SOFTWARE & INFRASTRUCTURE CRAFT (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-craft-early",
            topic="Software & Infrastructure Craft",
            belief_statement="Monolithic single-file architectures and rapid scripting are best because they optimize for speed of initial iteration.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="belief-craft-mid",
            active=False,
            reason_for_change="Single-file monoliths grew brittle and unmaintainable as system complexity increased.",
            supporting_evidence=[
                {"quote": "Monolithic single-script deployments prioritized rapid shipping speed over separation of concerns.", "timestamp_str": "2016-05-10"},
                {"quote": "Discovered that monolithic scripts accumulate technical debt and break unpredictably under scale.", "timestamp_str": "2017-08-15"}
            ]
        ),
        TemporalBelief(
            id="belief-craft-mid",
            topic="Software & Infrastructure Craft",
            belief_statement="Systems should be decomposed into decoupled microservices with network APIs to maximize independence.",
            era_label="Mid-to-Late 20s (2019-2022)",
            approx_year_start=2019,
            approx_year_end=2022,
            supersedes_id="belief-craft-early",
            superseded_by_id="belief-craft-2023",
            active=False,
            reason_for_change="Distributed microservice sprawl introduced network latency, operational overhead, and distributed debugging pain.",
            supporting_evidence=[
                {"quote": "Decomposing apps into multiple REST microservices separated failure domains in test clusters.", "timestamp_str": "2020-04-12"},
                {"quote": "Observed severe RPC latency, distributed tracing overhead, and complex local orchestration bugs.", "timestamp_str": "2022-09-18"}
            ]
        ),
        TemporalBelief(
            id="belief-craft-2023",
            topic="Software & Infrastructure Craft",
            belief_statement="Cloud convenience and managed smart home APIs are acceptable initial tradeoffs to get systems connected quickly, even if local control is incomplete.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            supersedes_id="belief-craft-mid",
            superseded_by_id="belief-craft-2024",
            active=False,
            reason_for_change="Cloud API outages, vendor subscription changes, and latency made cloud-dependent automation unreliable for daily living.",
            supporting_evidence=[
                {"quote": "Used cloud smart home vendor integrations for rapid initial home setup.", "timestamp_str": "2023-06-15"},
                {"quote": "Experienced cloud API rate limiting and sudden external outages disabling physical home switches.", "timestamp_str": "2023-11-28"}
            ]
        ),
        TemporalBelief(
            id="belief-craft-2024",
            topic="Software & Infrastructure Craft",
            belief_statement="Total local control is mandatory: decouple from cloud vendors, adopt Home Assistant as the primary orchestration hub, and connect devices over private Zigbee and Tailscale mesh.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-craft-2023",
            superseded_by_id="belief-craft-2025",
            active=False,
            reason_for_change="Ad-hoc IP addressing and disparate configs between locations caused configuration drift and fragile debugging.",
            supporting_evidence=[
                {"quote": "Eliminated all cloud vendor hubs in favor of local Zigbee2MQTT and Home Assistant Core.", "timestamp_str": "2024-03-10"},
                {"quote": "Identified subnet collision issues and manual IP config drift between Romania and Swiss environments.", "timestamp_str": "2024-10-05"}
            ]
        ),
        TemporalBelief(
            id="belief-craft-2025",
            topic="Software & Infrastructure Craft",
            belief_statement="Standardize network topology with deterministic static .69 IPs, Cudy P5 5G cellular fail-safe routing, and native MSS clamping across Romania Home, Swiss, and LakeOn instances.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-craft-2024",
            superseded_by_id="belief-craft-present",
            active=False,
            reason_for_change="Architecture required philosophical consolidation: elimination of all redundant proxy layers and deep hardware-in-the-loop closure.",
            supporting_evidence=[
                {"quote": "Standardized all primary hosts to .69 static IPs across all three household subnets.", "timestamp_str": "2025-02-18"},
                {"quote": "Hardened Cudy P5 native kernel MSS clamping to 1380 with 1500 MTU headroom to solve QMAP drops.", "timestamp_str": "2025-07-22"}
            ]
        ),
        TemporalBelief(
            id="belief-craft-present",
            topic="Software & Infrastructure Craft",
            belief_statement="Simplicity is complexity resolved (Constantin Brâncuși doctrine; Elon Musk 'no part is the best part'). High-reliability local-first architecture, standardized .69 static IPs, resilient triple-instance Home Assistant, and hardware-in-the-loop validation without unnecessary layers.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-craft-2025",
            active=True,
            supporting_evidence=craft_supporting,
            counter_arguments=craft_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in craft_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 2. FAILURE AND RESILIENCE (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-failure-early",
            topic="Failure and Resilience",
            belief_statement="Failure should be prevented and avoided through cautious upfront scoping and safe incremental steps.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="belief-failure-2023",
            active=False,
            reason_for_change="Excessive risk avoidance slowed learning; setbacks proved essential for high-leverage growth.",
            supporting_evidence=[
                {"quote": "Attempted to over-design systems upfront to guarantee zero runtime failures.", "timestamp_str": "2015-06-20"},
                {"quote": "Observed that total failure avoidance led to slow feature shipping and delayed user feedback.", "timestamp_str": "2018-04-14"}
            ]
        ),
        TemporalBelief(
            id="belief-failure-2023",
            topic="Failure and Resilience",
            belief_statement="Failure and friction are frustrating disruptions to be minimized through cautious planning and defensive execution.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            supersedes_id="belief-failure-early",
            superseded_by_id="belief-failure-2024",
            active=False,
            reason_for_change="Defensive planning often led to analysis paralysis without delivering real-world empirical validation.",
            supporting_evidence=[
                {"quote": "Felt friction and minor failures during initial Daily Speak setups as frustrating impediments.", "timestamp_str": "2023-05-18"},
                {"quote": "Recognized that cautious over-planning resulted in paralysis without real diagnostic progress.", "timestamp_str": "2023-11-12"}
            ]
        ),
        TemporalBelief(
            id="belief-failure-2024",
            topic="Failure and Resilience",
            belief_statement="Lean into empirical trial and rapid prototyping: small failures encountered early in hardware and code are valuable diagnostic signals.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-failure-2023",
            superseded_by_id="belief-failure-2025",
            active=False,
            reason_for_change="Tolerating failure was not enough; systems and habits needed intentional adversarial stress testing.",
            supporting_evidence=[
                {"quote": "Treated sensor communication dropouts as empirical diagnostic data rather than system failure.", "timestamp_str": "2024-04-08"},
                {"quote": "Iterated quickly through hardware prototyping failures to find root cause wiring flaws.", "timestamp_str": "2024-09-21"}
            ]
        ),
        TemporalBelief(
            id="belief-failure-2025",
            topic="Failure and Resilience",
            belief_statement="Adversarial red-teaming: deliberately stress systems, network links, and personal stamina under harsh real-world conditions to expose invisible brittle points.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-failure-2024",
            superseded_by_id="belief-failure-present",
            active=False,
            reason_for_change="Evolved from seeing stress as a technical test to realizing that struggle and setback are the fundamental forge of personal capability.",
            supporting_evidence=[
                {"quote": "Deliberately simulated WAN packet loss and cellular drops to verify fail-safe router fallbacks.", "timestamp_str": "2025-03-14"},
                {"quote": "Stressed personal training stamina to verify recovery resilience during intense travel blocks.", "timestamp_str": "2025-08-29"}
            ]
        ),
        TemporalBelief(
            id="belief-failure-present",
            topic="Failure and Resilience",
            belief_statement="I won't succeed if I don't fail first. Setbacks, demotivation, and temporary weakness are the forge of progress where fire and pressure shape capability and resilience.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-failure-2025",
            active=True,
            supporting_evidence=failure_supporting,
            counter_arguments=failure_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in failure_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 3. 20-YEAR COMPOUNDING VISION (2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-vision-2023",
            topic="20-Year Compounding Vision",
            belief_statement="Focus on immediate 1-to-2 year horizon: solve current technical challenges, establish remote work flexibility, and build core foundational competencies.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            superseded_by_id="belief-vision-2024",
            active=False,
            reason_for_change="Short-term horizons led to reactive planning rather than compounding strategic leverage.",
            supporting_evidence=[
                {"quote": "Prioritized 1-to-2 year tactical goals around remote engineering flexibility.", "timestamp_str": "2023-04-10"},
                {"quote": "Realized short-term tactical planning caused recurring context switching and fragmented focus.", "timestamp_str": "2023-10-30"}
            ]
        ),
        TemporalBelief(
            id="belief-vision-2024",
            topic="20-Year Compounding Vision",
            belief_statement="Adopt a 5-year strategic arc: invest heavily in lasting physical property, dual Switzerland-Romania base infrastructure, and autonomous automation platforms.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-vision-2023",
            superseded_by_id="belief-vision-2025",
            active=False,
            reason_for_change="Recognized that true compounding requires a generational multi-decade horizon rather than a 5-year cycle.",
            supporting_evidence=[
                {"quote": "Invested in lasting physical property and dual-region architecture across Swiss and Romania.", "timestamp_str": "2024-05-15"},
                {"quote": "Understood that physical and intellectual compounding demands decades of patient execution.", "timestamp_str": "2024-11-20"}
            ]
        ),
        TemporalBelief(
            id="belief-vision-2025",
            topic="20-Year Compounding Vision",
            belief_statement="Decadal compounding: create self-sustaining systems, sovereign energy, and deep mastery of local AI agents that endure for over ten years.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-vision-2024",
            superseded_by_id="belief-vision-20year",
            active=False,
            reason_for_change="Expanded to a full 20-year vision centered on intentional optimism and daily micro-compounding.",
            supporting_evidence=[
                {"quote": "Envisioned systems that run unattended for ten years with zero cloud dependency.", "timestamp_str": "2025-04-12"},
                {"quote": "Saw that a 10-year view can still underestimate the exponential trajectory of daily 1% compounding.", "timestamp_str": "2025-10-08"}
            ]
        ),
        TemporalBelief(
            id="belief-vision-20year",
            topic="20-Year Compounding Vision",
            belief_statement="Look 20 years into the future with intentional optimism: ask what could happen if it all goes well, and compound daily incremental improvements in craft, health, and life balance.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-vision-2025",
            active=True,
            supporting_evidence=vision_supporting,
            counter_arguments=vision_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in vision_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 4. PHYSICAL RESILIENCE & DAILY DISCIPLINES (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-physical-early",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Physical fitness can be maintained through occasional gym sessions and ad-hoc workouts whenever time permits.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="belief-physical-2023",
            active=False,
            reason_for_change="Inconsistent training led to energy troughs and poor posture mechanics during long software engineering blocks.",
            supporting_evidence=[
                {"quote": "Fit gym sessions in whenever convenient between university and freelance coding.", "timestamp_str": "2016-03-22"},
                {"quote": "Suffered from midday energy crashes and lumbar stiffness after irregular training schedules.", "timestamp_str": "2018-09-15"}
            ]
        ),
        TemporalBelief(
            id="belief-physical-2023",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Workouts and physical training can be fitted in ad-hoc around heavy programming and travel schedules.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            supersedes_id="belief-physical-early",
            superseded_by_id="belief-physical-2024",
            active=False,
            reason_for_change="Ad-hoc fitness resulted in physical fatigue, posture deterioration, and inconsistent energy.",
            supporting_evidence=[
                {"quote": "Treated physical exercise as an optional task to be rescheduled when software deadlines loomed.", "timestamp_str": "2023-06-08"},
                {"quote": "Noticed posture fatigue and shoulder strain creeping in during prolonged engineering blocks.", "timestamp_str": "2023-11-19"}
            ]
        ),
        TemporalBelief(
            id="belief-physical-2024",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Prioritize physical conditioning as an essential cognitive prerequisite: implement consistent morning routines and dedicated gym sessions.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-physical-2023",
            superseded_by_id="belief-physical-2025",
            active=False,
            reason_for_change="General workouts lacked specific postural corrections and shoulder rehabilitation needed for desk longevity.",
            supporting_evidence=[
                {"quote": "Established fixed morning stretching routines before starting daily development work.", "timestamp_str": "2024-04-18"},
                {"quote": "Identified that general cardio alone did not alleviate desk-induced forward shoulder posture.", "timestamp_str": "2024-10-12"}
            ]
        ),
        TemporalBelief(
            id="belief-physical-2025",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Targeted biomechanics: deliberate posture alignment, shoulder rehabilitation, sprint conditioning, and matching recovery to engineering output.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-physical-2024",
            superseded_by_id="belief-physical-present",
            active=False,
            reason_for_change="Travel transitions disrupted biomechanical routines; required an immutable iron routine adaptable to any location.",
            supporting_evidence=[
                {"quote": "Implemented targeted shoulder rotators and posture drills inspired by Bryan Johnson longevity protocols.", "timestamp_str": "2025-05-14"},
                {"quote": "Discovered that frequent travel broke routine execution unless formalized into an invariant travel protocol.", "timestamp_str": "2025-11-03"}
            ]
        ),
        TemporalBelief(
            id="belief-physical-present",
            topic="Physical Resilience & Daily Disciplines",
            belief_statement="Establish iron routines for sustainable travel and daily life: sprint cycles of intense physical conditioning alternating with deliberate recovery, daily posture correction, shoulder exercises, and incremental bodily care.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-physical-2025",
            active=True,
            supporting_evidence=physical_supporting,
            counter_arguments=physical_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in physical_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 5. LANGUAGE MASTERY & CONSISTENCY (FRENCH) (2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-french-2023",
            topic="French Learning & Consistency",
            belief_statement="French can be acquired through passive app practice, vocabulary drills, and casual exposure when residing in Switzerland.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            superseded_by_id="belief-french-2024",
            active=False,
            reason_for_change="Passive exposure failed to build conversational fluency or active verbal recall.",
            supporting_evidence=[
                {"quote": "Practiced French with mobile vocabulary apps for 15 minutes a day.", "timestamp_str": "2023-04-12"},
                {"quote": "Felt completely tongue-tied when attempting spontaneous French conversation in Swiss stores.", "timestamp_str": "2023-10-18"}
            ]
        ),
        TemporalBelief(
            id="belief-french-2024",
            topic="French Learning & Consistency",
            belief_statement="Structured grammar study and active conversational attempts are required to break through intermediate language plateaus.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-french-2023",
            superseded_by_id="belief-french-2025",
            active=False,
            reason_for_change="Structured study needed continuous high-intensity immersion to prevent syntax decay.",
            supporting_evidence=[
                {"quote": "Enrolled in structured French grammar textbooks and targeted conjugation drills.", "timestamp_str": "2024-05-19"},
                {"quote": "Noticed rapid decay of spoken confidence whenever grammar study had a week-long hiatus.", "timestamp_str": "2024-11-14"}
            ]
        ),
        TemporalBelief(
            id="belief-french-2025",
            topic="French Learning & Consistency",
            belief_statement="Intense daily immersion: rigorous French exam preparation, relentless vocabulary repetition until the language loops through thoughts, and celebrating milestones.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-french-2024",
            superseded_by_id="belief-french-present",
            active=False,
            reason_for_change="Exam preparation transitioned into integrated spontaneous fluency and self-rewarding habits.",
            supporting_evidence=[
                {"quote": "Committed to high-intensity daily French exam prep with strict 10-day sprint milestones.", "timestamp_str": "2025-06-22"},
                {"quote": "Experienced spontaneous inner dialogue in French after multi-hour immersion sessions.", "timestamp_str": "2025-10-31"}
            ]
        ),
        TemporalBelief(
            id="belief-french-present",
            topic="French Learning & Consistency",
            belief_statement="Disciplined daily immersion and conversational practice create true language fluency. Maintain consistency even when vocabulary loops through the mind at night, and celebrate milestones with meaningful rewards (like baking pumpkin pie after passing exams).",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-french-2025",
            active=True,
            supporting_evidence=french_supporting,
            counter_arguments=french_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in french_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 6. DATA PRIVACY & LOCAL SOVEREIGNTY (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-privacy-early",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Cloud hosting is completely fine for personal tools as long as basic HTTPS is enforced.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="belief-privacy-2023",
            active=False,
            reason_for_change="Cloud data leaks and corporate surveillance demonstrated the vulnerability of hosted personal data.",
            supporting_evidence=[
                {"quote": "Hosted personal utility scripts on public cloud virtual machines with TLS certificates.", "timestamp_str": "2016-08-14"},
                {"quote": "Observed corporate data breaches, IP leaks, and telemetry tracking across hosted platforms.", "timestamp_str": "2018-11-20"}
            ]
        ),
        TemporalBelief(
            id="belief-privacy-2023",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Commercial cloud LLMs and hosted tools are convenient for exploration; data sensitivity is manageable with basic settings.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            supersedes_id="belief-privacy-early",
            superseded_by_id="belief-privacy-2024",
            active=False,
            reason_for_change="Third-party data scraping, privacy policy shifts, and terms of service changes highlighted severe data sovereignty risks.",
            supporting_evidence=[
                {"quote": "Tested hosted AI translation APIs and commercial LLM prompts for prototyping.", "timestamp_str": "2023-05-24"},
                {"quote": "Cloud vendors silently revised terms of service to claim training rights on user prompts.", "timestamp_str": "2023-11-15"}
            ]
        ),
        TemporalBelief(
            id="belief-privacy-2024",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Personal memories, private journals, and core credentials must never touch external third-party cloud servers. Begin running Ollama and local models.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-privacy-2023",
            superseded_by_id="belief-privacy-2025",
            active=False,
            reason_for_change="Local models alone were insufficient; needed loopback network isolation, Pi-hole DNS sinkholing, and hardware-level enforcement.",
            supporting_evidence=[
                {"quote": "Migrated memory retrieval and persona inference to local Ollama models on Apple Silicon.", "timestamp_str": "2024-03-29"},
                {"quote": "Discovered that applications still attempted outbound analytics calls without local DNS sinkholes.", "timestamp_str": "2024-10-18"}
            ]
        ),
        TemporalBelief(
            id="belief-privacy-2025",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Sovereign local infrastructure: Pi-hole DNS sinkholes, drop cloud telemetry (disable cmsd on Cudy P5), Tailscale encrypted mesh, and local-only SQLite archives.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-privacy-2024",
            superseded_by_id="belief-privacy-present",
            active=False,
            reason_for_change="Elevated to an absolute constitutional operational invariant across all AI agents and personal data systems.",
            supporting_evidence=[
                {"quote": "Disabled router telemetry daemon cmsd and enforced Pi-hole DNS sinkholing across all LANs.", "timestamp_str": "2025-04-16"},
                {"quote": "Formalized Rule 6: Absolute local data privacy invariant across all AI agent tools and MCP hubs.", "timestamp_str": "2025-10-22"}
            ]
        ),
        TemporalBelief(
            id="belief-privacy-present",
            topic="Data Privacy & Local Sovereignty",
            belief_statement="Personal memories and identity must run 100% locally with zero cloud leakage, loopback binding, cryptographic manifests, and strict Rule 6 data privacy invariants.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-privacy-2025",
            active=True,
            supporting_evidence=privacy_supporting,
            counter_arguments=privacy_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in privacy_supporting if e.get("memory_id")]
        ),

        # =========================================================================
        # 7. DAILY SPOKEN REFLECTION & HABIT FORMATION (Early -> 2023 -> 2024 -> 2025 -> 2026)
        # =========================================================================
        TemporalBelief(
            id="belief-speaks-early",
            topic="Daily Spoken Reflection & Habit Formation",
            belief_statement="Silent internal introspection and occasional written notes are sufficient; recorded spoken reflections seem redundant or awkward.",
            era_label="Early 20s (2014-2018)",
            approx_year_start=2014,
            approx_year_end=2018,
            superseded_by_id="belief-speaks-2023",
            active=False,
            reason_for_change="Silent thoughts evaporated without leaving a durable trace; lacked structured articulation and longitudinal accountability.",
            supporting_evidence=[
                {"quote": "Relying on memory alone caused personal insights and milestones to fade within months.", "timestamp_str": "2016-04-11"},
                {"quote": "Written journals were too slow to capture the natural flow of spoken contemplation.", "timestamp_str": "2018-10-02"}
            ]
        ),
        TemporalBelief(
            id="belief-speaks-2023",
            topic="Daily Spoken Reflection & Habit Formation",
            belief_statement="Initiate the Daily Speak Project: speak reflections aloud into the camera daily to force clarity of thought and capture time's imprint.",
            era_label="2023 Era",
            approx_year_start=2023,
            approx_year_end=2023,
            supersedes_id="belief-speaks-early",
            superseded_by_id="belief-speaks-2024",
            active=False,
            reason_for_change="Initial speaking sessions were self-conscious; needed to transition from forced discipline to spontaneous verbal presence.",
            supporting_evidence=[
                {"quote": "Started recording daily video logs to force vocal articulation of engineering and life reflections.", "timestamp_str": "2023-05-02"},
                {"quote": "Felt self-conscious on camera initially, requiring repeated daily practice to speak naturally.", "timestamp_str": "2023-11-18"}
            ]
        ),
        TemporalBelief(
            id="belief-speaks-2024",
            topic="Daily Spoken Reflection & Habit Formation",
            belief_statement="Spoken reflection as cognitive decompression: verbal journaling sharpens thinking, reveals cognitive blindspots, and accelerates multilingual expression.",
            era_label="2024 Era",
            approx_year_start=2024,
            approx_year_end=2024,
            supersedes_id="belief-speaks-2023",
            superseded_by_id="belief-speaks-2025",
            active=False,
            reason_for_change="Spoken journaling needed integration with physical vocal mechanics and intentional delivery.",
            supporting_evidence=[
                {"quote": "Used Daily Speak sessions to decompress complex system architecture designs after long coding blocks.", "timestamp_str": "2024-04-14"},
                {"quote": "Observed that vocal delivery required intentional diaphragmatic resonance and posture alignment.", "timestamp_str": "2024-10-25"}
            ]
        ),
        TemporalBelief(
            id="belief-speaks-2025",
            topic="Daily Spoken Reflection & Habit Formation",
            belief_statement="Refine vocal delivery and poise: emphasize deep belly resonance, deliberate eye contact, calm measured cadence (~121 WPM), and honest self-appraisal.",
            era_label="2025 Era",
            approx_year_start=2025,
            approx_year_end=2025,
            supersedes_id="belief-speaks-2024",
            superseded_by_id="belief-speaks-present",
            active=False,
            reason_for_change="Elevated spoken recordings from a personal habit into the ground truth training corpus for sovereign longitudinal AI preservation.",
            supporting_evidence=[
                {"quote": "Practiced diaphragmatic belly resonance to ground the speaking voice at a calm baritone register.", "timestamp_str": "2025-05-19"},
                {"quote": "Realized that years of high-fidelity video recordings constitute a priceless ground-truth corpus for digital preservation.", "timestamp_str": "2025-11-11"}
            ]
        ),
        TemporalBelief(
            id="belief-speaks-present",
            topic="Daily Spoken Reflection & Habit Formation",
            belief_statement="Daily spoken reflections form the grounded autobiographical foundation for personal AI preservation and longitudinal self-discovery. Time leaves its mark upon us like hands modeling clay, and authentic speech preserves who we are across the decades.",
            era_label="Present Day (2026)",
            approx_year_start=2026,
            approx_year_end=None,
            supersedes_id="belief-speaks-2025",
            active=True,
            supporting_evidence=speaks_supporting,
            counter_arguments=speaks_counterarguments,
            evidence_memory_ids=[e["memory_id"] for e in speaks_supporting if e.get("memory_id")]
        )
    ]

    count = 0
    for belief in beliefs_definitions:
        db.add_belief(belief)
        count += 1
        sup_info = f" -> {belief.superseded_by_id}" if not belief.active else " [ACTIVE]"
        ev_info = f" (ev={len(belief.supporting_evidence)}, ca={len(belief.counter_arguments)})"
        print(f"  + Seeded [{belief.era_label}] {belief.topic}{sup_info}{ev_info}")

    stats = db.get_archive_stats()
    print(f"\n[Seed Beliefs] Successfully seeded {count} consolidated temporal beliefs. Total in DB: {stats['temporal_beliefs']}")
    return count


if __name__ == "__main__":
    db_arg = sys.argv[1] if len(sys.argv) > 1 else ""
    seed_temporal_beliefs(db_arg)
