"""schemas_observer_decider.py

Zbiór schematów Pydantic opisujących strukturę JSON-ów, które
chcemy wymusić od modelu językowego (Gemini / OpenAI) w zadaniu
Observer ⇄ Decider.

Każda klasa:
• posiada atrybut klasowy `prompt_text` – gotowy prompt do wysłania;
• określa ścisły kształt oczekiwanego JSON-a (Literal / bool / str).

Użycie:
    from schemas_observer_decider import ClusterThemeSchema
    data = ClusterThemeSchema.model_validate({"cluster_theme": "Video setup"})
"""
from __future__ import annotations

from typing import Literal, ClassVar

from pydantic import BaseModel, Field


class ObserverDeciderSchema(BaseModel):
    """Klasa bazowa; podklasy definiują własne pola wyjściowe
    i nadpisują `prompt_text`.
    """

    # prompt_text to metadane – NIE może być polem modelu
    prompt_text: ClassVar[str]

    model_config = {
        "extra": "forbid",          # blokuj nieznane klucze
        "populate_by_name": True,
    }


# ============================================================================
# Schematy – każda klasa ma własny prompt i pola wyniku
# ============================================================================

class AbstractConcreteSchema(ObserverDeciderSchema):
    abstract_vs_concrete: Literal["abstract", "concrete"]
    prompt_text: ClassVar[str] = (
        "Is the language predominantly abstract-conceptual or concrete-sensory?\n"
        'Return { "abstract_vs_concrete":"abstract" }.'
    )


class ClusterThemeSchema(ObserverDeciderSchema):
    cluster_theme: str = Field(...)
    prompt_text: ClassVar[str] = (
        "Give a 1-3 word topic label summarising the concrete subject of the rant "
        '(e.g. "father-absence", "Excel-overload").\n\n'
        "Respond **only** with this exact JSON object – nothing more, nothing less, "
        'and use straight ASCII double quotes (").\n\n'
        '{"cluster_theme":"<your-label-here>"}'
    )


class CoffeeShopFlipSchema(ObserverDeciderSchema):
    coffee_shop_flip: bool
    prompt_text: ClassVar[str] = (
        "If this is a LOW-INTENSITY segment, does it talk calmly about the opposite "
        "domain compared to the main spikes? Yes = coffee-shop baseline.\n"
        'Return { "coffee_shop_flip":false }.'
    )


class DecisionFocusSchema(ObserverDeciderSchema):
    dec_focus: Literal["self", "tribe"]
    prompt_text: ClassVar[str] = (
        "For a PEOPLE segment: is the emotional spike aimed at SELF (my feelings, my worth) "
        "or TRIBE (they judge me, their needs)?\n"
        'Return { "dec_focus":"self" }.'
    )


class EmoRootSchema(ObserverDeciderSchema):
    emo_root: Literal["fear", "pain"]
    prompt_text: ClassVar[str] = (
        "Assuming the segment was labelled T or P, decide whether the dominant emotional "
        "root is FEAR (anxiety, uncertainty) or PAIN (hurt, judgment).\n"
        'Return { "emo_root":"fear" }.'
    )


class FreakoutTypeSchema(ObserverDeciderSchema):
    freakout_type: str = Field(...)
    prompt_text: ClassVar[str] = (
        "Map the core rant to the closest OPS ‘Choose Your Freak-out’ archetype "
        "(e.g. ‘lost-keys’, ‘barbecue-judgement’, ‘calendar-collapse’).\n"
        'Return { "freakout_type":"lost-keys" }.'
    )


class MetaFlagSchema(ObserverDeciderSchema):
    meta_flag: bool
    prompt_text: ClassVar[str] = (
        "Is the speaker only narrating the process (e.g. ‘I’m recording’, book titles, "
        "tech setup) without emotional content?\n"
        'Return { "meta_flag":true }.'
    )


class MissingInfoLoopSchema(ObserverDeciderSchema):
    missing_info_loop: bool
    prompt_text: ClassVar[str] = (
        "Does the speaker repeat phrases about hidden information, conspiracy or ‘missing pieces’? "
        'If yes, return { "missing_info_loop":true } else false.'
    )


class ObsessionFocusSchema(ObserverDeciderSchema):
    obs_focus: Literal["chaos", "control"]
    prompt_text: ClassVar[str] = (
        "For a THINGS segment: does the speaker obsess over missing options & flexibility (chaos) "
        "or over enforcing plans & limits (control)?\n"
        'Return { "obs_focus":"chaos" }.'
    )


class ODTagSchema(ObserverDeciderSchema):
    OD_tag: Literal["T", "P", "M"]
    prompt_text: ClassVar[str] = (
        "Read the segment below. If the main emotional topic is about THINGS/INFORMATION/CHAOS-CONTROL label T; "
        "if it is about PEOPLE/RELATIONSHIPS/SELF-TRIBE label P; if it is meta, purely descriptive or has no "
        'emotional charge, label M.\nReturn JSON → { "OD_tag": "T" }.'
    )


class OrganizerGathererSchema(ObserverDeciderSchema):
    organizer_gatherer: Literal["organizer", "gatherer"]
    prompt_text: ClassVar[str] = (
        "If THINGS: classify whether the rant tries to lock things down (organizer) "
        "or to keep options open (gatherer).\n"
        'Return { "organizer_gatherer":"gatherer" }.'
    )


class PainFearLexScoreSchema(ObserverDeciderSchema):
    pain_fear_lex_score: Literal["high", "medium", "low"]
    prompt_text: ClassVar[str] = (
        "Count explicit pain-words (hurt, unfair, rejected) vs fear-words (chaos, lost, uncertain). "
        "If either list ≥ 4 tokens, mark ‘high’; 2-3 → ‘medium’; else ‘low’.\n"
        'Return { "pain_fear_lex_score":"high" }.'
    )


class TriggerTxtSchema(ObserverDeciderSchema):
    trigger_txt: str = Field(...)
    prompt_text: ClassVar[str] = (
        "Extract the first 3–6 words that immediately precede the emotional spike "
        "(highest volume or strongest wording). Quote exactly.\n"
        'Return { "trigger_txt":"I almost have no…" }.'
    )


class ConsumptionBalanceSchema(ObserverDeciderSchema):
    consume_vs_blast: Literal["consume", "blast"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily struggle with reluctance to share known information (stuck/consume) "
        "or with over-sharing/teaching known information to the tribe quickly (blast)?\n"
        '{ "consume_vs_blast":"consume" }'
    )


class InfoFlowDirectionSchema(ObserverDeciderSchema):
    flow_direction: Literal["gathering_for_self", "sharing_known_for_tribe"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main drive to **gather new knowledge and experiences for self-absorption** (gathering_for_self), "
        "or to **take known information and quickly share/teach it to the Tribe** (sharing_known_for_tribe)?\n"
        '{ "flow_direction":"gathering_for_self" }'
    )


class ReadinessStanceSchema(ObserverDeciderSchema):
    readiness_stance: Literal["never_ready_perfectionism", "confidently_ready_rush"]
    prompt_text: ClassVar[str] = (
        "Does the speaker constantly feel **unprepared or stuck, requiring one more book or class before starting** (never_ready_perfectionism), "
        "or do they exhibit **overconfidence, immediately sharing or acting on limited knowledge** (confidently_ready_rush)?\n"
        '{ "readiness_stance":"never_ready_perfectionism" }'
    )


class DemonAnimalAvoidanceSchema(ObserverDeciderSchema):
    avoidance_focus: Literal["avoids_new_research", "avoids_sharing_teaching"]
    prompt_text: ClassVar[str] = (
        "Does the speaker actively **avoid looking up new information, doing deep research, or reading one more book** (avoids_new_research), "
        "or do they actively **resist communicating plans, sharing knowledge, or getting started with others** (avoids_sharing_teaching)?\n"
        '{ "avoidance_focus":"avoids_sharing_teaching" }'
    )


class InfoUpdateGoalSchema(ObserverDeciderSchema):
    info_goal: Literal["updates_inner_worldview", "reinforces_current_worldview"]
    prompt_text: ClassVar[str] = (
        "When encountering new information, is the goal to **use it to refine or potentially crash and update the inner worldview** (updates_inner_worldview), "
        "or to **force the new data to conform and reinforce the current established worldview** (reinforces_current_worldview)?\n"
        '{ "info_goal":"updates_inner_worldview" }'
    )


class ActionInitiationSchema(ObserverDeciderSchema):
    initiation_method: Literal["start_by_setting_low_bar", "start_by_teaching_plan"]
    prompt_text: ClassVar[str] = (
        "To start a major task, does the speaker intentionally **set an extremely low, non-committal bar to trick themselves into moving** (start_by_setting_low_bar), "
        "or do they start by **immediately teaching, explaining, or presenting the gathered information/plan to the Tribe** (start_by_teaching_plan)?\n"
        '{ "initiation_method":"start_by_setting_low_bar" }'
    )


# ---------------------------------------------------------------------------
# Animals: Sleep vs Play schemas
# ---------------------------------------------------------------------------


class EnergyPrioritySchema(ObserverDeciderSchema):
    energy_focus: Literal["preserve_energy_first", "expend_energy_first"]
    prompt_text: ClassVar[str] = (
        "Does the speaker prioritize **preserving energy and processing internally** before acting (preserve_energy_first), "
        "or **expending energy externally through chaotic action and interaction** (expend_energy_first)?\n"
        '{ "energy_focus":"preserve_energy_first" }'
    )


class LearningMethodSchema(ObserverDeciderSchema):
    learning_method: Literal["internal_processing_before_doing", "random_learning_via_chaos"]
    prompt_text: ClassVar[str] = (
        "Does the speaker's learning strategy favor **deep internal processing and time to prepare** (internal_processing_before_doing), "
        "or **random, chaotic engagement with the external world and learning through trial-and-error with the Tribe** (random_learning_via_chaos)?\n"
        '{ "learning_method":"internal_processing_before_doing" }'
    )


class ExhaustionResponseSchema(ObserverDeciderSchema):
    exhaustion_nature: Literal["mentally_drained_need_isolation", "physically_drained_need_rest"]
    prompt_text: ClassVar[str] = (
        "When tired, does the speaker experience primary exhaustion as **mental drain, requiring complete isolation to recharge the inner world** (mentally_drained_need_isolation), "
        "or as **physical exhaustion from external activity, while the mind remains energized** (physically_drained_need_rest)?\n"
        '{ "exhaustion_nature":"mentally_drained_need_isolation" }'
    )


class ChaosToleranceSchema(ObserverDeciderSchema):
    chaos_tolerance: Literal["seeks_external_chaos_interaction", "avoids_external_chaos_interaction"]
    prompt_text: ClassVar[str] = (
        "Is the speaker comfortable or energized by **randomly encountering, dealing with, and maneuvering through external chaos and social interaction** (seeks_external_chaos_interaction), "
        "or do they actively **avoid high-energy, chaotic social demands** to maintain internal preparedness (avoids_external_chaos_interaction)?\n"
        '{ "chaos_tolerance":"seeks_external_chaos_interaction" }'
    )


class ActionInitiationEnergySchema(ObserverDeciderSchema):
    initiation_method: Literal["needs_long_time_to_process_ready", "jumps_in_with_both_feet"]
    prompt_text: ClassVar[str] = (
        "Does the speaker communicate a need for a **long processing time to feel ready, often leading to getting stuck** (needs_long_time_to_process_ready), "
        "or do they exhibit a readiness to **jump into tasks and start moving immediately, despite the chaos** (jumps_in_with_both_feet)?\n"
        '{ "initiation_method":"needs_long_time_to_process_ready" }'
    )


class StressCopingMechanismSchema(ObserverDeciderSchema):
    coping_mechanism: Literal["crash_from_overworking_battery_dies", "retreat_from_overwhelming_play"]
    prompt_text: ClassVar[str] = (
        "When under extreme stress, does the speaker experience a catastrophic **crash, feeling sick, or physical breakdown due to prolonged energy expenditure** (crash_from_overworking_battery_dies), "
        "or does it manifest as a **mental retreat or isolation from an overwhelming external social environment** (retreat_from_overwhelming_play)?\n"
        '{ "coping_mechanism":"crash_from_overworking_battery_dies" }'
    )

class SelfWorthSourceSchema(ObserverDeciderSchema):
    worth_source: Literal["worth_from_internal_preparedness", "worth_from_external_action"]
    prompt_text: ClassVar[str] = (
        "Does the speaker derive self-worth primarily from **having preserved energy, being internally prepared, and having processes settled** (worth_from_internal_preparedness), "
        "or from **extroverted movement, chaotic learning, and navigating external interactions** (worth_from_external_action)?\n"
        '{ "worth_source":"worth_from_internal_preparedness" }'
    )


class NoveltyHandlingSchema(ObserverDeciderSchema):
    novelty_approach: Literal["new_things_are_energy_drain", "new_things_are_energy_source"]
    prompt_text: ClassVar[str] = (
        "Does the speaker feel that **unplanned novelty and chaotic external events are draining and disruptive** to their necessary processing time (new_things_are_energy_drain), "
        "or that **random external interactions and new, unexpected scenarios provide necessary energy and learning** (new_things_are_energy_source)?\n"
        '{ "novelty_approach":"new_things_are_energy_drain" }'
    )


class StucknessMechanismSchema(ObserverDeciderSchema):
    stuckness_mechanism: Literal["stuck_due_to_unprocessed_self", "stuck_due_to_lack_of_external_action"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's feeling of being 'stuck' caused by **not having enough time to process or prepare internally** (stuck_due_to_unprocessed_self), "
        "or by **avoiding the chaotic, physical, and random external action/interaction required to break the inertia** (stuck_due_to_lack_of_external_action)?\n"
        '{ "stuckness_mechanism":"stuck_due_to_unprocessed_self" }'
    )


class MomentumStrategySchema(ObserverDeciderSchema):
    momentum_strategy: Literal["achieve_momentum_via_internal_push", "achieve_momentum_via_external_chaos"]
    prompt_text: ClassVar[str] = (
        "To gain momentum, does the speaker rely on **internal discipline, deep processing, and slow, meticulous progress** (achieve_momentum_via_internal_push), "
        "or on **spontaneous action, quick trial-and-error, and seeking out energetic, chaotic external input** (achieve_momentum_via_external_chaos)?\n"
        '{ "momentum_strategy":"achieve_momentum_via_external_chaos" }'
    )


class WorkEnduranceSchema(ObserverDeciderSchema):
    endurance_limit: Literal["sustained_internal_focus", "short_bursts_of_external_energy"]
    prompt_text: ClassVar[str] = (
        "Does the speaker exhibit the ability for **long periods of sustained internal focus, processing, and preparation** (sustained_internal_focus), "
        "or do they tend to operate in **short, high-energy bursts of external activity and communication** before tiring (short_bursts_of_external_energy)?\n"
        '{ "endurance_limit":"sustained_internal_focus" }'
    )

class ChaosInteractionValueSchema(ObserverDeciderSchema):
    chaos_value: Literal["chaotic_action_as_fuel", "chaotic_action_as_drain"]
    prompt_text: ClassVar[str] = (
        "Does the speaker view spontaneous, chaotic external action and interaction with the Tribe as **necessary fuel for learning and progress** (chaotic_action_as_fuel), "
        "or as an **unwanted energy drain that interferes with internal preparation** (chaotic_action_as_drain)?\n"
        '{ "chaos_value":"chaotic_action_as_drain" }'
    )


class WorkBoundarySchema(ObserverDeciderSchema):
    work_boundary: Literal["difficulty_stopping_working", "difficulty_starting_working"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main struggle the **inability to stop working, rest, or pause processing**, often leading to physical or mental exhaustion** (difficulty_stopping_working), "
        "or is it the **inability to start tasks, constantly demanding more time for preparation** (difficulty_starting_working)?\n"
        '{ "work_boundary":"difficulty_starting_working" }'
    )


class RechargeMethodSchema(ObserverDeciderSchema):
    recharge_method: Literal["recharge_via_alone_time", "recharge_via_extroverted_activity"]
    prompt_text: ClassVar[str] = (
        "When feeling drained, does the speaker seek **isolation, withdrawal, and quiet internal processing to recharge** (recharge_via_alone_time), "
        "or do they seek **extroverted interaction, physical activity, and chaotic social engagement** (recharge_via_extroverted_activity)?\n"
        '{ "recharge_method":"recharge_via_alone_time" }'
    )


class InternalProcessingSchema(ObserverDeciderSchema):
    processing_style: Literal["internal_dialogue_processing", "external_pinging_processing"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily process thoughts and problems through **deep internal reflection and dialogue** (internal_dialogue_processing), "
        "or by **externalizing ideas and 'pinging' them off the Tribe** (external_pinging_processing)?\n"
        '{ "processing_style":"internal_dialogue_processing" }'
    )


class HealthAwarenessSchema(ObserverDeciderSchema):
    health_awareness: Literal["hyper_aware_of_owies_and_sickness", "ignores_physical_exhaustion"]
    prompt_text: ClassVar[str] = (
        "Is the speaker highly conscious and often vocal about their **sickness, tiredness, or the need to preserve energy** (hyper_aware_of_owies_and_sickness), "
        "or do they tend to **power through physical exhaustion, often crashing unexpectedly** (ignores_physical_exhaustion)?\n"
        '{ "health_awareness":"hyper_aware_of_owies_and_sickness" }'
    )


# ---------------------------------------------------------------------------
# IntroExtro (Introversion vs Extroversion) schemas
# ---------------------------------------------------------------------------


class DominantFunctionDirectionSchema(ObserverDeciderSchema):
    dominant_direction: Literal["savior_functions_are_introverted", "savior_functions_are_extroverted"]
    prompt_text: ClassVar[str] = (
        "Are the speaker's dominant mental functions primarily focused on **internal worlds, processing, and self-reference (Introverted)** (savior_functions_are_introverted), "
        "or on **external data gathering, objective systems, and engagement with the Tribe (Extroverted)** (savior_functions_are_extroverted)?\n"
        '{ "dominant_direction":"savior_functions_are_introverted" }'
    )


class ConflictCopingDirectionSchema(ObserverDeciderSchema):
    coping_direction: Literal["escape_to_inner_world", "push_outward_to_solve"]
    prompt_text: ClassVar[str] = (
        "When faced with major life problems or stress, is the speaker's default reaction to **retreat, isolate, and focus on self-processing** (escape_to_inner_world), "
        "or to **push the energy outward, engage externally, and seek an action-based solution with the Tribe** (push_outward_to_solve)?\n"
        '{ "coping_direction":"escape_to_inner_world" }'
    )


class EnergyRechargeSchema(ObserverDeciderSchema):
    recharge_method: Literal["recharge_via_isolation_alone_time", "recharge_via_tribe_interaction"]
    prompt_text: ClassVar[str] = (
        "Does the speaker feel the need to **withdraw from the external world and spend time alone** to recover energy (recharge_via_isolation_alone_time), "
        "or do they feel energized by **engaging with others, external activity, and Tribe interaction** (recharge_via_tribe_interaction)?\n"
        '{ "recharge_method":"recharge_via_isolation_alone_time" }'
    )


class CommunicationGoalIntroExtroSchema(ObserverDeciderSchema):
    communication_goal: Literal["communication_as_self_defense", "communication_as_tribe_feedback"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's external communication primarily motivated by **protecting their inner world/identity or minimizing external exposure** (communication_as_self_defense), "
        "or by **actively seeking feedback, validation, or engagement from the Tribe** (communication_as_tribe_feedback)?\n"
        '{ "communication_goal":"communication_as_tribe_feedback" }'
    )


class AttentionSeekingMethodSchema(ObserverDeciderSchema):
    attention_method: Literal["attention_through_controlled_output", "attention_through_extroverted_display"]
    prompt_text: ClassVar[str] = (
        "Does the speaker seek recognition and attention primarily through **highly prepared, self-contained products, systems, or achievements** (attention_through_controlled_output), "
        "or through **immediate, energetic, and often chaotic external engagement with the Tribe** (attention_through_extroverted_display)?\n"
        '{ "attention_method":"attention_through_controlled_output" }'
    )


# ---------------------------------------------------------------------------
# Animals: Blast vs Play and Sleep vs Consume schemas
# ---------------------------------------------------------------------------


class ExtrovertedExpressionFocusSchema(ObserverDeciderSchema):
    expression_focus: Literal["overconfident_sharing_of_limited_knowledge", "physical_action_over_verbal_sharing"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main output characterized by **quickly communicating and teaching known information to the Tribe** (overconfident_sharing_of_limited_knowledge), "
        "or by **chaotic, spontaneous physical action and learning through trial-and-error interaction** (physical_action_over_verbal_sharing)?\n"
        '{ "expression_focus":"overconfident_sharing_of_limited_knowledge" }'
    )


class TensionReleaseMethodSchema(ObserverDeciderSchema):
    tension_release: Literal["tension_released_through_verbal_blast", "tension_released_through_physical_play"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily relieve internal tension by **talking, teaching, and communicating their knowledge/reasons** (tension_released_through_verbal_blast), "
        "or by **engaging in physical movement, chaotic external actions, and learning through doing with others** (tension_released_through_physical_play)?\n"
        '{ "tension_release":"tension_released_through_verbal_blast" }'
    )


class TribeCorrectionStanceSchema(ObserverDeciderSchema):
    correction_stance: Literal["actively_corrects_the_tribe", "seeks_spontaneous_movement"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's external energy frequently directed at **aggressively correcting the Tribe's logic or values** (actively_corrects_the_tribe), "
        "or primarily at **seeking out opportunities for spontaneous, chaotic, and physical interaction/activity** (seeks_spontaneous_movement)?\n"
        '{ "correction_stance":"actively_corrects_the_tribe" }'
    )


class PerfectionismFocusSchema(ObserverDeciderSchema):
    perfectionism_focus: Literal["perfectionism_of_internal_processing", "perfectionism_of_gathering_enough"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's perfectionism rooted in ensuring **the internal system is fully processed and prepared** (perfectionism_of_internal_processing), "
        "or in the anxiety of **never having gathered enough new information or experiences** before acting (perfectionism_of_gathering_enough)?\n"
        '{ "perfectionism_focus":"perfectionism_of_gathering_enough" }'
    )


class ActionInertiaSchema(ObserverDeciderSchema):
    inertia_point: Literal["difficulty_stopping_consuming", "difficulty_starting_sleep"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main struggle the **inability to stop taking in new information** (difficulty_stopping_consuming), "
        "or the **inability to start or stop internal processing/rest** (difficulty_starting_sleep)?\n"
        '{ "inertia_point":"difficulty_stopping_consuming" }'
    )


class InformationAnxietySchema(ObserverDeciderSchema):
    anxiety_type: Literal["anxiety_over_unseen_options", "anxiety_over_unprepared_system"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's anxiety rooted in the fear of **missing crucial external information or options** (anxiety_over_unseen_options), "
        "or the fear of **a system breakdown due to lack of internal organization or preparation** (anxiety_over_unprepared_system)?\n"
        '{ "anxiety_type":"anxiety_over_unprepared_system" }'
    )


class EnergyDeploymentSchema(ObserverDeciderSchema):
    sleep_vs_play: Literal["sleep", "play"]
    prompt_text: ClassVar[str] = (
        "When faced with a challenge, does the speaker's default reaction lead to withdrawal for internal processing/readying (sleep), "
        "or jumping into action/extroverted chaos and maneuvering with the tribe (play)?\n"
        '{ "sleep_vs_play":"sleep" }'
    )


class DeciderModalitySchema(ObserverDeciderSchema):
    decider_push: Literal["push_self", "push_tribe"]
    prompt_text: ClassVar[str] = (
        "Is the emotional certainty/push primarily directed at defending their internal identity/reasons (Masculine DI), "
        "or is it directed at forcing actions/validation onto the Tribe (Masculine DE)?\n"
        '{ "decider_push":"push_self" }'
    )


class ObserverModalitySchema(ObserverDeciderSchema):
    observer_certainty: Literal["solid_facts", "solid_concepts"]
    prompt_text: ClassVar[str] = (
        "Is the observer's demand for certainty focused on unmovable, specific Facts, Data, or Details (Masculine Sensory), "
        "or on solid, intricate Conceptual Frameworks and Patterns (Masculine Intuition)?\n"
        '{ "observer_certainty":"solid_facts" }'
    )


class FocusDominanceSchema(ObserverDeciderSchema):
    dominant_focus: Literal["info_dominant", "energy_dominant"]
    prompt_text: ClassVar[str] = (
        "Does the speaker prioritize gathering knowledge/information/understanding (Info Dominant) "
        "or prioritize external action, movement, and quick deployment of energy (Energy Dominant)?\n"
        '{ "dominant_focus":"info_dominant" }'
    )


# ---------------------------------------------------------------------------
# Additional critical Observer/Decider binary schemas
# ---------------------------------------------------------------------------


class ObjectOfSpikeSchema(ObserverDeciderSchema):
    spike_object: Literal["things_system", "person_character"]
    prompt_text: ClassVar[str] = (
        "During the emotional spike, is the frustration directed primarily at non-human objects, systems, "
        "or information processes (things_system), or at the character, name, or identity of specific people or groups (person_character)?\n"
        '{ "spike_object":"things_system" }'
    )


class UltimateFearSchema(ObserverDeciderSchema):
    ultimate_fear: Literal["control_trickery", "judgment_exile"]
    prompt_text: ClassVar[str] = (
        "Does the speaker’s life imbalance center around the anxiety of being deceived, controlled, or lacking crucial information (control_trickery), "
        "or around the pain of being judged, exiled, or losing personal worth (judgment_exile)?\n"
        '{ "ultimate_fear":"control_trickery" }'
    )


class CoreMovabilitySchema(ObserverDeciderSchema):
    core_movability: Literal["people_movable", "things_movable"]
    prompt_text: ClassVar[str] = (
        "In the speaker’s perspective, are people and social dynamics generally easy to navigate or change (people_movable), "
        "or are non-human elements (facts, systems, information) viewed as easily adaptable or movable (things_movable)? "
        "(Choose the area perceived as easiest to move or change).\n"
        '{ "core_movability":"people_movable" }'
    )


class MiddleFunctionFocusSchema(ObserverDeciderSchema):
    calm_focus_area: Literal["talks_calmly_things", "talks_calmly_people", "calm_things", "calm_people"]
    prompt_text: ClassVar[str] = (
        "In low-intensity segments, is the speaker relatively calm and objective when discussing external information/systems (talks_calmly_things or calm_things), "
        "or when discussing people, identity and values (talks_calmly_people or calm_people)?\n"
        '{ "calm_focus_area":"talks_calmly_things" }'
    )


class CoreParalysisSchema(ObserverDeciderSchema):
    core_paralysis: Literal["stuck_by_things", "stuck_by_people_identity"]
    prompt_text: ClassVar[str] = (
        "Which domain causes the most intense, life-debilitating feeling of being absolutely 'stuck': "
        "non-human elements, systems, or missing information (stuck_by_things), "
        "or personal identity, worth, or being judged/exiled by the Tribe (stuck_by_people_identity)?\n"
        '{ "core_paralysis":"stuck_by_things" }'
    )


class CommunicationGoalSchema(ObserverDeciderSchema):
    communication_goal: Literal["seek_validation_acceptance", "seek_clarity_proof"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core goal in their communication or actions to achieve external acceptance/validation from others (seek_validation_acceptance), "
        "or to establish precise facts, proof, and clarity about a system or reality (seek_clarity_proof)?\n"
        '{ "communication_goal":"seek_clarity_proof" }'
    )

class EmotionsRoleSchema(ObserverDeciderSchema):
    emotions_role: Literal["primary_value", "secondary_reason"]
    prompt_text: ClassVar[str] = (
        "In the decision-making process, are feelings and values used as the primary filter/authority (primary_value), "
        "or must emotions be suppressed and logically justified as a secondary check after logic/reasons are established (secondary_reason)?\n"
        '{ "emotions_role":"secondary_reason" }'
    )


class ParalysisSourceSchema(ObserverDeciderSchema):
    paralysis_source: Literal["stuck_by_systems", "stuck_by_identity"]
    prompt_text: ClassVar[str] = (
        "Does the speaker express an irrational, 'forever fear' of being permanently stuck due to external things, broken systems, or missing data (stuck_by_systems), "
        "or due to a permanent state of being judged, rejected, or worthless to the Tribe (stuck_by_identity)?\n"
        '{ "paralysis_source":"stuck_by_systems" }'
    )


class UltimateGoalBehaviorSchema(ObserverDeciderSchema):
    ultimate_goal: Literal["functional_clarity", "emotional_alignment"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core goal to achieve clarity, proof, accuracy, or ensure a system/thing works correctly (functional_clarity), "
        "or to achieve social acceptance, emotional connection, or alignment with personal/group values (emotional_alignment)?\n"
        '{ "ultimate_goal":"functional_clarity" }'
    )


class ConflictAdaptabilitySchema(ObserverDeciderSchema):
    conflict_adaptability: Literal["people_negotiable", "facts_negotiable"]
    prompt_text: ClassVar[str] = (
        "When encountering inevitable conflict, is the speaker more capable of maneuvering and adapting through complexities involving people and social dynamics (people_negotiable), "
        "or through complexities involving facts, data, and system logic (facts_negotiable)?\n"
        '{ "conflict_adaptability":"facts_negotiable" }'
    )


class ClarityDemandSchema(ObserverDeciderSchema):
    clarity_demand: Literal["demands_clarity", "accepts_guessing"]
    prompt_text: ClassVar[str] = (
        "Does the speaker insistently demand absolute clarity, factual accuracy, and precise rules from the external world (demands_clarity), "
        "or are they comfortable with making educated guesses, forming patterns, and proceeding without perfect factual clarity (accepts_guessing)?\n"
        '{ "clarity_demand":"demands_clarity" }'
    )


class HelpSeekingPatternSchema(ObserverDeciderSchema):
    help_focus: Literal["help_with_people", "help_with_pathway"]
    prompt_text: ClassVar[str] = (
        "When asking for external assistance, is the request primarily focused on figuring out the motivations, judgments, or social dynamics of people/Tribe (help_with_people), "
        "or is it focused on finding a clear method, pathway, concept, or missing information/fact (help_with_pathway)?\n"
        '{ "help_focus":"help_with_pathway" }'
    )

class SpikeObjectDetailSchema(ObserverDeciderSchema):
    spike_object_detail: Literal["things_system_malfunction", "people_character_flaw"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's emotional spike intensely focused on the failure of a non-human system (e.g., faulty plug, printer, or administrative process) "
        "or on the character, flaws, or actions of specific individuals or the Tribe (people_character_flaw)?\n"
        '{ "spike_object_detail":"things_system_malfunction" }'
    )


class PrimaryFearSourceSchema(ObserverDeciderSchema):
    fear_source: Literal["lack_of_information", "lack_of_worth"]
    prompt_text: ClassVar[str] = (
        "Is the dominant anxiety rooted in the idea of crucial knowledge being hidden, information being incomplete, or being tricked by reality (lack_of_information), "
        "or is it rooted in the fear of being judged, rejected, or losing significance/value in the eyes of the Tribe or Self (lack_of_worth)?\n"
        '{ "fear_source":"lack_of_information" }'
    )


class ImbalanceConsequenceSchema(ObserverDeciderSchema):
    imbalance_consequence: Literal["uncontrolled_chaos_control", "uncontrolled_judgment_exile"]
    prompt_text: ClassVar[str] = (
        "Is the overarching life problem centered on the struggle to manage chaotic information or restrictive control of external systems (uncontrolled_chaos_control), "
        "or on dealing with continuous social conflict, pain from judgment, or fear of exile (uncontrolled_judgment_exile)?\n"
        '{ "imbalance_consequence":"uncontrolled_chaos_control" }'
    )


class SelfBlameFocusSchema(ObserverDeciderSchema):
    self_blame: Literal["failing_to_observe", "failing_to_decide"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily self-criticize for 'not gathering enough facts,' 'not being prepared,' or 'missing the plan' (failing_to_observe), "
        "or for 'losing control of emotions,' 'being judgmental,' or 'not caring enough about others' (failing_to_decide)?\n"
        '{ "self_blame":"failing_to_observe" }'
    )


class MessyToleranceSchema(ObserverDeciderSchema):
    mess_tolerance: Literal["triggered_by_messy_things", "calm_about_messy_things"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's frustration heavily directed at physical disarray, system disorganization, or general lack of factual clarity (triggered_by_messy_things), "
        "or is this aspect relatively ignored or accepted in favor of focusing on people/identity issues (calm_about_messy_things)?\n"
        '{ "mess_tolerance":"triggered_by_messy_things" }'
    )


class PersonalOffenseSchema(ObserverDeciderSchema):
    personal_offense_trigger: Literal["thing_chaos_personal_offense", "tribe_rejection_personal_offense"]
    prompt_text: ClassVar[str] = (
        "Is the deepest emotional trigger a personally offensive reaction to broken systems, physical chaos, or information failure (thing_chaos_personal_offense), "
        "or is it a personally offensive reaction to feeling judged, rejected, or unappreciated by others/the Tribe (tribe_rejection_personal_offense)?\n"
        '{ "personal_offense_trigger":"thing_chaos_personal_offense" }'
    )


class PrimaryPackMotivationSchema(ObserverDeciderSchema):
    pack_motivation: Literal["acceptance_via_doing_it_right", "acceptance_via_straight_shot"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's drive for acceptance primarily focused on ensuring clarity, proving competence, and getting the thing right (acceptance_via_doing_it_right), "
        "or is it focused directly on seeking validation, significance, or resolving interpersonal/identity issues (acceptance_via_straight_shot)?\n"
        '{ "pack_motivation":"acceptance_via_doing_it_right" }'
    )


class MagicShowSyndromeSchema(ObserverDeciderSchema):
    magic_show_focus: Literal["sees_magic_show_in_things", "sees_magic_show_in_people"]
    prompt_text: ClassVar[str] = (
        "Does the speaker's emotional instability manifest as being 'trapped in a continual magic show' concerning objects, systems, or missing information (sees_magic_show_in_things), "
        "or as struggling to grasp the motivations and consistency of the Tribe (sees_magic_show_in_people)?\n"
        '{ "magic_show_focus":"sees_magic_show_in_things" }'
    )


class FunctionalLagTimeSchema(BaseModel):
    lag_area: Literal["lag_on_new_info", "lag_on_emotional_reasons"]
    summary: Optional[str] = Field(None, description="Optional explanation text")
    evidence: Optional[List[str]] = Field(default=None, description="Optional evidence quotes")

    prompt_text: ClassVar[str] = (
        "Does the speaker exhibit noticeable 'lag time' or difficulty when processing and accepting new external information or experiences "
        "(lag_on_new_info), or when expressing or justifying emotions, values, or logical reasons for a decision (lag_on_emotional_reasons)?\n"
        '{ "lag_area": "lag_on_new_info" }'
    )

    class Config:
        extra = "ignore"

class DeciderPushOriginSchema(ObserverDeciderSchema):
    push_origin: Literal["push_internal_identity", "push_external_tribe_norms"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's dominant, masculine certainty (shove) focused on defending or imposing their internal, subjective truth or value (push_internal_identity), "
        "or on establishing, challenging, or dictating the external reasons or values of the Tribe/System (push_external_tribe_norms)?\n"
        '{ "push_origin":"push_external_tribe_norms" }'
    )


class OpinionRejectionSchema(ObserverDeciderSchema):
    rejection_reaction: Literal["unmovable_on_self_reasons", "unmovable_on_tribe_acceptance"]
    prompt_text: ClassVar[str] = (
        "When challenged, is the speaker's defensive response to assert their own personal subjective standards, logic, or values as unmovable (unmovable_on_self_reasons), "
        "or to assert external standards of fairness, logic, or necessity for the Tribe as unmovable (unmovable_on_tribe_acceptance)?\n"
        '{ "rejection_reaction":"unmovable_on_tribe_acceptance" }'
    )


class DeciderValueSourceSchema(ObserverDeciderSchema):
    value_source: Literal["significance_over_validation", "validation_over_significance"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's primary drive for self-worth based on achieving personal significance or self-importance (significance_over_validation), "
        "or is it based on acquiring external validation, approval, or acceptance from the Tribe (validation_over_significance)?\n"
        '{ "value_source":"validation_over_significance" }'
    )


class NegativeExpressionModalitySchema(ObserverDeciderSchema):
    negative_expression: Literal["aggressive_punchy_to_tribe", "internal_bitchy_complainy"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's primary method of expressing strong negative emotion outward through aggressive, punching, or direct confrontation/correction of the Tribe (aggressive_punchy_to_tribe), "
        "or is it characterized by whiny, complaining, or 'bitchy' external emotional leakage focused on their own frustration/insecurity (internal_bitchy_complainy)?\n"
        '{ "negative_expression":"internal_bitchy_complainy" }'
    )

class AccommodationFocusSchema(ObserverDeciderSchema):
    accommodation_focus: Literal["tribe_must_be_accommodated", "self_must_be_accommodated"]
    prompt_text: ClassVar[str] = (
        "In social interaction or conflict, does the speaker prioritize adjusting their external behavior, words, or emotions to fit the Tribe's standards (tribe_must_be_accommodated), "
        "or do they strongly assert that the Tribe must recognize and adapt to their internal identity or values (self_must_be_accommodated)?\n"
        '{ "accommodation_focus":"tribe_must_be_accommodated" }'
    )


class NegativeExpressionFocusSchema(ObserverDeciderSchema):
    negative_expression_focus: Literal["aggressive_punch_to_tribe", "whiny_internal_complain"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's negative emotional output primarily characterized by aggressive shoves, direct confrontation, or corrections aimed at the Tribe (aggressive_punch_to_tribe), "
        "or is it manifested as whiny, complaining, or bitchy leakage focused on internal frustration or insecurity (whiny_internal_complain)?\n"
        '{ "negative_expression_focus":"aggressive_punch_to_tribe" }'
    )


class TribeIdentityBeliefSchema(ObserverDeciderSchema):
    tribe_identity_belief: Literal["trusts_tribe_identity", "skeptical_of_tribe_identity"]
    prompt_text: ClassVar[str] = (
        "Does the speaker exhibit a default tendency to assume the best about the Tribe's intentions, identities, and values (trusts_tribe_identity), "
        "or are they naturally skeptical, wary, or immediately seeking flaws in the Tribe's identity and morality (skeptical_of_tribe_identity)?\n"
        '{ "tribe_identity_belief":"skeptical_of_tribe_identity" }'
    )


class SocialNicetiesPrioritySchema(ObserverDeciderSchema):
    social_niceties_priority: Literal["social_niceties_essential", "social_niceties_bullshit"]
    prompt_text: ClassVar[str] = (
        "Does the speaker place a conscious or unconscious emphasis on maintaining social harmony, emotional politeness, or using formalized pleasantries (social_niceties_essential), "
        "or do they tend to disregard or actively reject these niceties in favour of direct, often aggressive confrontation or honesty (social_niceties_bullshit)?\n"
        '{ "social_niceties_priority":"social_niceties_bullshit" }'
    )

class PrimaryValidationGoalSchema(ObserverDeciderSchema):
    validation_goal: Literal["significance_over_validation", "validation_over_significance"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main drive for self-worth rooted in achieving personal significance or self-importance (significance_over_validation), "
        "or in earning external validation, approval, or acceptance from the Tribe (validation_over_significance)?\n"
        '{ "validation_goal":"significance_over_validation" }'
    )


class DeciderShoveDirectionSchema(ObserverDeciderSchema):
    shove_direction: Literal["push_on_self", "push_on_tribe"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's most intense, demanding energy (shove) primarily directed at their own internal standards, identity, or discipline (push_on_self), "
        "or at challenging, correcting, or organizing the reasons/values of others (push_on_tribe)?\n"
        '{ "shove_direction":"push_on_tribe" }'
    )


class TribeObligationSenseSchema(ObserverDeciderSchema):
    tribe_obligation: Literal["obligated_to_answer", "prioritizes_self_privacy"]
    prompt_text: ClassVar[str] = (
        "Does the speaker communicate a sense of social obligation to respond, engage, or satisfy the Tribe's inquiries/demands (obligated_to_answer), "
        "or do they quickly assert personal boundaries and prioritize internal privacy/needs (prioritizes_self_privacy)?\n"
        '{ "tribe_obligation":"prioritizes_self_privacy" }'
    )


class TribeClarityPerceptionSchema(ObserverDeciderSchema):
    tribe_clarity: Literal["tribe_is_blurry_fuzzy", "tribe_is_trustworthy_authority"]
    prompt_text: ClassVar[str] = (
        "In moments of uncertainty, does the speaker view the Tribe’s identity, values, or reasons as inconsistent, blurry, or difficult to grasp (tribe_is_blurry_fuzzy), "
        "or do they tend to treat the Tribe's opinions, feelings, or consensus as a reliable source of external authority (tribe_is_trustworthy_authority)?\n"
        '{ "tribe_clarity":"tribe_is_blurry_fuzzy" }'
    )


class AchievementPrideSourceSchema(ObserverDeciderSchema):
    pride_source: Literal["pride_in_earned_character", "pride_in_tribe_recognition"]
    prompt_text: ClassVar[str] = (
        "When discussing personal success, is the speaker’s emphasis placed on internal character, effort, or self-development that earned the achievement (pride_in_earned_character), "
        "or on the external recognition, praise, or validation received from others/the system (pride_in_tribe_recognition)?\n"
        '{ "pride_source":"pride_in_earned_character" }'
    )


class DemonProcessingMethodSchema(ObserverDeciderSchema):
    processing_method: Literal["process_via_external_vomiting", "process_via_internal_isolation"]
    prompt_text: ClassVar[str] = (
        "When dealing with emotional or identity turmoil, does the speaker tend to resolve it by extrovertedly engaging, talking things out, or seeking external feedback/interaction (process_via_external_vomiting), "
        "or by withdrawing into an internal world, wrestling with the self, and resisting external intrusion (process_via_internal_isolation)?\n"
        '{ "processing_method":"process_via_internal_isolation" }'
    )


class DeciderMovabilitySchema(ObserverDeciderSchema):
    movable_area: Literal["tribe_is_movable", "self_is_movable"]
    prompt_text: ClassVar[str] = (
        "In conflicts related to values or identity, does the speaker prioritize changing or correcting the Tribe's external standards/expectations (tribe_is_movable), "
        "or do they prioritize changing or adjusting their own self/identity/expression to match external conditions (self_is_movable)?\n"
        '{ "movable_area":"tribe_is_movable" }'
    )


class SignificanceStrategySchema(ObserverDeciderSchema):
    significance_method: Literal["significance_via_solo_achievement", "significance_via_helping_others"]
    prompt_text: ClassVar[str] = (
        "Does the speaker communicate self-worth primarily through personal, independent accomplishments, knowledge, or identity building (significance_via_solo_achievement), "
        "or through external validation earned by helping, engaging, or satisfying the Tribe's needs (significance_via_helping_others)?\n"
        '{ "significance_method":"significance_via_solo_achievement" }'
    )


class IdentityDisclosureSchema(ObserverDeciderSchema):
    identity_disclosure: Literal["disclosure_only_if_forced", "disclosure_as_default_interaction"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily reveal deep personal thoughts, struggles, or identity issues only when pressured, challenged, or in controlled private settings (disclosure_only_if_forced), "
        "or do they spontaneously share and vent emotional and identity states as a default method of external engagement (disclosure_as_default_interaction)?\n"
        '{ "identity_disclosure":"disclosure_only_if_forced" }'
    )


class TribeExpectationReactionSchema(ObserverDeciderSchema):
    expectation_reaction: Literal["resist_tribe_expectations", "overextend_for_tribe_expectations"]
    prompt_text: ClassVar[str] = (
        "When faced with external social pressure, does the speaker's default reaction involve resisting, pushing back, and establishing boundaries against those expectations (resist_tribe_expectations), "
        "or does it involve overextending and trying to meet or live up to all external demands (overextend_for_tribe_expectations)?\n"
        '{ "expectation_reaction":"overextend_for_tribe_expectations" }'
    )


class EmotionalVentingMethodSchema(ObserverDeciderSchema):
    venting_method: Literal["systematic_venting", "organic_spontaneous_venting"]
    prompt_text: ClassVar[str] = (
        "When dealing with demon emotions (anger, sadness), does the speaker try to find a structured, logical, or systematic outlet (systematic_venting), "
        "or is the emotional release spontaneous, chaotic, and organically integrated with daily interactions and self-expression (organic_spontaneous_venting)?\n"
        '{ "venting_method":"systematic_venting" }'
    )


class DeepRootedFrustrationSchema(ObserverDeciderSchema):
    core_frustration: Literal["printer_paperwork", "judgment_rejection"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's deepest, most persistent, and most offensive frustration directed at malfunctions, missing information, or administrative systems (printer_paperwork), "
        "or at personal judgment, character flaws of others, and feeling rejected or exiled by the group (judgment_rejection)?\n"
        '{ "core_frustration":"printer_paperwork" }'
    )


class AreaOfFlexibilitySchema(ObserverDeciderSchema):
    easy_to_change: Literal["things_flexible", "people_flexible"]
    prompt_text: ClassVar[str] = (
        "In the speaker’s spontaneous communication (baseline talk), which domain is treated as easily adaptable, fixable, or generally 'not a huge problem': "
        "external systems, processes, or facts (things_flexible), or social dynamics and relationships (people_flexible)?\n"
        '{ "easy_to_change":"things_flexible" }'
    )


class FearOfExileSchema(ObserverDeciderSchema):
    exile_source: Literal["tricked_by_system", "worthless_exiled_by_tribe"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core fear rooted in being manipulated, lied to, or controlled by external systems or information (tricked_by_system), "
        "or is it rooted in losing personal worth, being judged, or being exiled/rejected by the Tribe (worthless_exiled_by_tribe)?\n"
        '{ "exile_source":"worthless_exiled_by_tribe" }'
    )


class UltimateGoalSchema(ObserverDeciderSchema):
    ultimate_goal: Literal["get_it_working_right", "get_tribe_acceptance"]
    prompt_text: ClassVar[str] = (
        "Is the speaker primarily striving to perfect a system, acquire definitive proof, or ensure clarity in reality (get_it_working_right), "
        "or to obtain emotional validation, acceptance, or alignment with the Tribe (get_tribe_acceptance)?\n"
        '{ "ultimate_goal":"get_tribe_acceptance" }'
    )


class SourceOfAuthoritySchema(ObserverDeciderSchema):
    primary_authority: Literal["facts_data", "feelings_values"]
    prompt_text: ClassVar[str] = (
        "Is the speaker’s final argument or point of authority based primarily on observable evidence, external data, or factual accuracy (facts_data), "
        "or on internal priorities, personal values, or emotional states (feelings_values)?\n"
        '{ "primary_authority":"facts_data" }'
    )

class DeciderShoveFocusSchema(ObserverDeciderSchema):
    shove_focus: Literal["internal_self_discipline", "external_tribe_correction"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's intense push/shove primarily directed at **their own internal standards, identity development, or discipline** (internal_self_discipline), "
        "or at **challenging, correcting, or organizing the Tribe's external reasons or values** (external_tribe_correction)?\n"
        '{ "shove_focus":"internal_self_discipline" }'
    )


class InsecuritySourceSchema(ObserverDeciderSchema):
    insecurity_root: Literal["insecurity_of_not_being_significant", "insecurity_of_not_being_validated"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core insecurity linked to a fear of **losing self-worth, not being important, or failing personal standards** (insecurity_of_not_being_significant), "
        "or to a fear of **not receiving external approval, validation, or acceptance from the Tribe** (insecurity_of_not_being_validated)?\n"
        '{ "insecurity_root":"insecurity_of_not_being_significant" }'
    )


class ProblemProcessingMethodSchema(ObserverDeciderSchema):
    processing_method: Literal["process_via_isolation_withdrawal", "process_via_venting_interaction"]
    prompt_text: ClassVar[str] = (
        "When faced with emotional or identity turmoil, does the speaker's default coping mechanism involve **withdrawing for internal wrestling and self-processing** (process_via_isolation_withdrawal), "
        "or **extrovertedly engaging, talking out, or venting emotions to external entities** (process_via_venting_interaction)?\n"
        '{ "processing_method":"process_via_isolation_withdrawal" }'
    )


class TribeExpectationStanceSchema(ObserverDeciderSchema):
    tribe_stance: Literal["resist_tribe_authority", "overextend_to_tribe_authority"]
    prompt_text: ClassVar[str] = (
        "Does the speaker's default stance toward external social pressure involve **resistance, assertion of personal autonomy, and challenging the Tribe's right to dictate** (resist_tribe_authority), "
        "or **an overextension of effort to meet or exceed external expectations and demands** (overextend_to_tribe_authority)?\n"
        '{ "tribe_stance":"resist_tribe_authority" }'
    )


class AchievementPrideFocusSchema(ObserverDeciderSchema):
    pride_focus: Literal["pride_in_unseen_effort", "pride_in_public_accolades"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's deepest pride associated with **internal self-discipline, consistently upholding personal values, or unseen efforts** (pride_in_unseen_effort), "
        "or with **receiving public recognition, awards, or direct validation from others** (pride_in_public_accolades)?\n"
        '{ "pride_focus":"pride_in_unseen_effort" }'
    )


class PrimaryLimitationFearSchema(ObserverDeciderSchema):
    limitation_fear: Literal["fear_of_control_restriction", "fear_of_chaos_unpreparedness"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's emotional spike rooted in anxiety about **external control, being limited, or being denied options** (fear_of_control_restriction), "
        "or in the terror of **uncontrolled chaos, unpredictable 'waves,' or lack of internal preparedness** (fear_of_chaos_unpreparedness)?\n"
        'Return { "limitation_fear":"fear_of_chaos_unpreparedness" }.'
    )


class InformationPrioritySchema(ObserverDeciderSchema):
    info_priority: Literal["prioritizes_known_information", "prioritizes_new_information"]
    prompt_text: ClassVar[str] = (
        "In handling data, does the speaker prioritize **organizing, refining, and relying on existing/known information** (prioritizes_known_information), "
        "or constantly **seeking out new data, options, and experiences** (prioritizes_new_information)?\n"
        'Return { "info_priority":"prioritizes_new_information" }.'
    )


class PlanAdherenceSchema(ObserverDeciderSchema):
    plan_adherence: Literal["rigid_adherence_to_plan", "willingness_to_break_plan"]
    prompt_text: ClassVar[str] = (
        "Does the speaker exhibit a tendency towards **rigid adherence to established plans, routines, or systems** (rigid_adherence_to_plan), "
        "or a tendency towards **spontaneous flexibility and a willingness to break routine or change paths** (willingness_to_break_plan)?\n"
        'Return { "plan_adherence":"willingness_to_break_plan" }.'
    )


class GuessingToleranceSchema(ObserverDeciderSchema):
    guessing_tolerance: Literal["guessing_is_uncomfortable", "guessing_is_autopilot"]
    prompt_text: ClassVar[str] = (
        "In situations lacking full information, does the speaker display **discomfort, anxiety, or reluctance to rely on intuition/guessing** (guessing_is_uncomfortable), "
        "or does making quick assumptions and relying on patterns/impulses feel like **autopilot or a natural way to proceed** (guessing_is_autopilot)?\n"
        'Return { "guessing_tolerance":"guessing_is_autopilot" }.'
    )


class StucknessNatureSchema(ObserverDeciderSchema):
    stuckness_nature: Literal["paralysis_due_to_missing_path", "paralysis_due_to_limiting_options"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's feeling of being 'stuck' primarily caused by **lacking a single clear plan, organized system, or defined pathway** (paralysis_due_to_missing_path), "
        "or by **not having enough alternative options, experiences, or external freedom to explore** (paralysis_due_to_limiting_options)?\n"
        'Return { "stuckness_nature":"paralysis_due_to_limiting_options" }.'
    )


class ProblemSolvingMethodSchema(ObserverDeciderSchema):
    problem_method: Literal["builds_brick_by_brick", "breaks_structure_to_find_solution"]
    prompt_text: ClassVar[str] = (
        "Does the speaker approach new tasks by focusing on **perfecting fundamental steps and building systematically** (builds_brick_by_brick), "
        "or by **throwing things at the wall, aggressively trying different methods, and breaking structures** to find a working solution (breaks_structure_to_find_solution)?\n"
        'Return { "problem_method":"breaks_structure_to_find_solution" }.'
    )


class FutureThreatFocusSchema(ObserverDeciderSchema):
    threat_focus: Literal["fear_of_known_tidal_wave", "fear_of_unknown_missing_info"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's anxiety about the future focused on **managing an anticipated, recognizable 'tidal wave' or system collapse** (fear_of_known_tidal_wave), "
        "or on the concern that **crucial information or options are currently missing/unseen, leading to inevitable failure** (fear_of_unknown_missing_info)?\n"
        'Return { "threat_focus":"fear_of_unknown_missing_info" }.'
    )

class NewInfoResistanceSchema(ObserverDeciderSchema):
    new_info_stance: Literal["resists_new_info_and_change", "constantly_seeks_new_info"]
    prompt_text: ClassVar[str] = (
        "Does the speaker exhibit resistance and anxiety towards the constant intake of new information or changes, prioritizing the known and stable (resists_new_info_and_change), "
        "or do they communicate an insatiable need for more new data, options, and experiences (constantly_seeks_new_info)?\n"
        'Return { "new_info_stance":"resists_new_info_and_change" }.'
    )


class PerfectionismFocusSchema(ObserverDeciderSchema):
    perfectionism_focus: Literal["perfect_internal_system", "perfect_external_options"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's perfectionism concentrated on **building, refining, and adhering to one single, flawless internal plan or system** (perfect_internal_system), "
        "or on **acquiring a complete and exhaustive spectrum of new options or information** (perfect_external_options)?\n"
        'Return { "perfectionism_focus":"perfect_internal_system" }.'
    )


class ActionStanceSchema(ObserverDeciderSchema):
    action_stance: Literal["needs_preparation_before_action", "jumps_in_to_chaos_mode"]
    prompt_text: ClassVar[str] = (
        "When faced with a difficult task, does the speaker tend to spend significant time **processing internally and preparing** before taking external action (needs_preparation_before_action), "
        "or do they **jump into the task chaotically, relying on improvisation and learning through external trial and error** (jumps_in_to_chaos_mode)?\n"
        'Return { "action_stance":"jumps_in_to_chaos_mode" }.'
    )


class RoutineStanceSchema(ObserverDeciderSchema):
    routine_stance: Literal["embraces_routine_repetition", "fears_routine_boredom"]
    prompt_text: ClassVar[str] = (
        "Does the speaker value and return to **routine, established processes, and repetitive tasks** (embraces_routine_repetition), "
        "or do they express anxiety about **boredom and the constant need for novel experiments and changes** (fears_routine_boredom)?\n"
        'Return { "routine_stance":"fears_routine_boredom" }.'
    )


class InformationProcessingGoalSchema(ObserverDeciderSchema):
    processing_goal: Literal["drive_to_narrow_down", "drive_to_expand_options"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main goal in analyzing information to **reduce variables, refine the core concept, and define one clear path** (drive_to_narrow_down), "
        "or to **collect as many data points, possibilities, or alternative ideas as possible** (drive_to_expand_options)?\n"
        'Return { "processing_goal":"drive_to_narrow_down" }.'
    )

class NewInfoApproachSchema(ObserverDeciderSchema):
    new_info_approach: Literal["active_search_for_new_info", "avoidance_of_new_info_search"]
    prompt_text: ClassVar[str] = (
        "Does the speaker consistently engage in the **active and restless search for new data, options, or experiences** (active_search_for_new_info), "
        "or do they express **resistance, dread, or avoidance when forced to seek out information they don't already possess** (avoidance_of_new_info_search)?\n"
        "Return { \"new_info_approach\":\"active_search_for_new_info\" }."
    )


class ProcessTrustSchema(ObserverDeciderSchema):
    process_trust: Literal["trust_in_systematic_knowns", "trust_in_chaotic_exploration"]
    prompt_text: ClassVar[str] = (
        "Does the speaker place primary trust in **systematic procedures, rigorous preparation, and known processes** (trust_in_systematic_knowns), "
        "or in **fast improvisation, chaotic trial-and-error, and spontaneous learning through external movement** (trust_in_chaotic_exploration)?\n"
        "Return { \"process_trust\":\"trust_in_chaotic_exploration\" }."
    )


class FactCheckingRoleSchema(ObserverDeciderSchema):
    fact_role: Literal["facts_used_for_clarity_and_conclusion", "facts_used_for_exploration_spectrum"]
    prompt_text: ClassVar[str] = (
        "Is the speaker utilizing facts/data primarily to **narrow down options and confirm a singular, accurate conclusion/plan** (facts_used_for_clarity_and_conclusion), "
        "or to **map out the widest possible range of options and gather details across the spectrum** (facts_used_for_exploration_spectrum)?\n"
        "Return { \"fact_role\":\"facts_used_for_exploration_spectrum\" }."
    )


class FreedomRestrictionRageSchema(ObserverDeciderSchema):
    restriction_rage: Literal["rage_at_external_restrictions", "rage_at_internal_unpreparedness"]
    prompt_text: ClassVar[str] = (
        "Is the emotional rage triggered by **external rules, control, or restrictions that limit freedom and options** (rage_at_external_restrictions), "
        "or by **internal failures in planning, organization, or lack of preparation for anticipated chaos** (rage_at_internal_unpreparedness)?\n"
        "Return { \"restriction_rage\":\"rage_at_external_restrictions\" }."
    )


class WorryHorizonSchema(ObserverDeciderSchema):
    worry_horizon: Literal["worry_about_organized_future_plans", "worry_about_unseen_current_options"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's long-term worry focused on **organizing and preparing for future, semi-predictable threats or systemic collapse** (worry_about_organized_future_plans), "
        "or on **the immediate anxiety of missing out on crucial options or unseen information in the present moment** (worry_about_unseen_current_options)?\n"
        "Return { \"worry_horizon\":\"worry_about_unseen_current_options\" }."
    )

class InformationResponsibilitySchema(ObserverDeciderSchema):
    responsibility_area: Literal["responsible_for_proof_and_facts", "responsible_for_patterns_and_concepts"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core focus and area of assumed responsibility in communication to establish **factual accuracy, details, and concrete proof** (responsible_for_proof_and_facts), "
        "or to identify **underlying patterns, conceptual frameworks, and abstract connections** (responsible_for_patterns_and_concepts)?\n"
        "Return { \"responsibility_area\":\"responsible_for_patterns_and_concepts\" }."
    )


class ClarityToleranceSchema(ObserverDeciderSchema):
    clarity_level: Literal["demands_clarity_and_literalism", "tolerates_vagueness_and_interpretation"]
    prompt_text: ClassVar[str] = (
        "Does the speaker strongly demand **literal, detailed, and clear communication/information** from the world (demands_clarity_and_literalism), "
        "or do they easily **tolerate and utilize abstract language, vague concepts, and room for interpretation** (tolerates_vagueness_and_interpretation)?\n"
        "Return { \"clarity_level\":\"tolerates_vagueness_and_interpretation\" }."
    )


class SubjectMatterFocusSchema(ObserverDeciderSchema):
    subject_focus: Literal["focuses_on_the_what_and_data", "focuses_on_the_concept_and_meaning"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's natural inclination to focus the conversation on **concrete facts, physical details, and quantifiable data ('the what')** (focuses_on_the_what_and_data), "
        "or on **abstract concepts, underlying meanings, and patterns ('the why/how it relates')** (focuses_on_the_concept_and_meaning)?\n"
        "Return { \"subject_focus\":\"focuses_on_the_concept_and_meaning\" }."
    )


class GuessingApproachSchema(ObserverDeciderSchema):
    guessing_approach: Literal["guessing_is_natural_default", "guessing_is_uncomfortable_without_facts"]
    prompt_text: ClassVar[str] = (
        "In situations with incomplete data, does the speaker rely heavily on **quick pattern recognition and intuitive leaps/guesses** (guessing_is_natural_default), "
        "or do they express **discomfort or avoidance when asked to guess without a foundation of concrete facts** (guessing_is_uncomfortable_without_facts)?\n"
        "Return { \"guessing_approach\":\"guessing_is_uncomfortable_without_facts\" }."
    )


class WorryFocusSchema(ObserverDeciderSchema):
    worry_focus: Literal["anxiety_over_present_physical_reality", "anxiety_over_future_patterns_systems"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core observational anxiety directed at **current, tangible, physical details, facts, or immediate events** (anxiety_over_present_physical_reality), "
        "or at **future conceptual threats, collapsing abstract patterns, or long-term systemic failures** (anxiety_over_future_patterns_systems)?\n"
        "Return { \"worry_focus\":\"anxiety_over_future_patterns_systems\" }."
    )

class DecisionFilterSchema(ObserverDeciderSchema):
    core_filter: Literal["logic_reasons_analysis", "values_prioritization_worth"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily filter decisions based on **logic, cause-and-effect reasoning, and analysis** (logic_reasons_analysis), "
        "or through **values, prioritizing importance, and assessing worth** (values_prioritization_worth)?\n"
        "Return { \"core_filter\":\"logic_reasons_analysis\" }."
    )


class ActionGoalSchema(ObserverDeciderSchema):
    action_goal: Literal["getting_it_done_working", "getting_it_valued_prioritized"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's main drive to ensure **the thing works, the logic fits, and the task is accomplished** (getting_it_done_working), "
        "or to **evaluate the worth, establish the pecking order, and prioritize value** (getting_it_valued_prioritized)?\n"
        "Return { \"action_goal\":\"getting_it_done_working\" }."
    )


class EmotionalLagSchema(ObserverDeciderSchema):
    emotion_timing: Literal["emotions_come_later", "emotions_are_immediate_filter"]
    prompt_text: ClassVar[str] = (
        "In stressful situations, do emotions and feelings typically **follow the completion of the task or the establishment of logic** (emotions_come_later), "
        "or are they **immediately present, acting as the primary filter** for the situation (emotions_are_immediate_filter)?\n"
        "Return { \"emotion_timing\":\"emotions_come_later\" }."
    )


class ConflictResponseSchema(ObserverDeciderSchema):
    conflict_response: Literal["logic_explains_failing_to_emote", "focuses_on_vibe_and_connection"]
    prompt_text: ClassVar[str] = (
        "In public/social conflict, does the speaker tend to respond by **over-explaining their logical reasons, missing the emotional plea** (logic_explains_failing_to_emote), "
        "or by **immediately focusing on the emotional atmosphere, social impact, and connection** (focuses_on_vibe_and_connection)?\n"
        "Return { \"conflict_response\":\"logic_explains_failing_to_emote\" }."
    )


class ArgumentationStanceSchema(ObserverDeciderSchema):
    argument_stance: Literal["argues_for_the_sake_of_reasons", "avoids_arguments_unless_values_hit"]
    prompt_text: ClassVar[str] = (
        "Does the speaker readily engage in **debates and arguments for the pleasure of honing logic and reasons** (argues_for_the_sake_of_reasons), "
        "or do they generally **avoid argumentation unless their core values or priorities are directly challenged** (avoids_arguments_unless_values_hit)?\n"
        "Return { \"argument_stance\":\"argues_for_the_sake_of_reasons\" }."
    )

class ThinkingMethodSchema(ObserverDeciderSchema):
    thinking_method: Literal["internal_thought_lab", "external_trial_and_error"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily solve logistical problems by **staring and thinking internally for the optimal way** (internal_thought_lab), "
        "or by **immediately trying different external actions and using trial-and-error** (external_trial_and_error)?\n"
        "Return { \"thinking_method\":\"external_trial_and_error\" }."
    )


class LogicFocusSchema(ObserverDeciderSchema):
    logic_focus: Literal["truth_for_self", "truth_for_tribe_efficiency"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's core logical drive to establish **subjective, personally consistent truths and internal systems** (truth_for_self), "
        "or to establish **objective, externally verified reasons that work efficiently for the Tribe/System** (truth_for_tribe_efficiency)?\n"
        "Return { \"logic_focus\":\"truth_for_tribe_efficiency\" }."
    )


class AdviceStanceSchema(ObserverDeciderSchema):
    advice_stance: Literal["advice_is_suggestion", "advice_is_order"]
    prompt_text: ClassVar[str] = (
        "When offering solutions or advice, does the speaker present it as a **suggestion based on a subjective viewpoint** (advice_is_suggestion), "
        "or as an **objective command or established best practice** (advice_is_order)?\n"
        "Return { \"advice_stance\":\"advice_is_order\" }."
    )


class ValueSourceSchema(ObserverDeciderSchema):
    value_source: Literal["self_values_and_beliefs", "social_spectrum_and_validation"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's value system primarily defined by **internal, individual feelings, beliefs, and morals** (self_values_and_beliefs), "
        "or by **external responses, the social spectrum, and the emotions of others** (social_spectrum_and_validation)?\n"
        "Return { \"value_source\":\"social_spectrum_and_validation\" }."
    )


class EmotionalExpressionSchema(ObserverDeciderSchema):
    emotional_expression: Literal["internal_intense_feelings", "external_emotional_vibe"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's emotional energy typically characterized by **intense, deeply personal feelings often kept internal or focused on self-identity** (internal_intense_feelings), "
        "or by **constant external monitoring and management of the emotional atmosphere/vibe of the group** (external_emotional_vibe)?\n"
        "Return { \"emotional_expression\":\"external_emotional_vibe\" }."
    )


class SocialHarmonySchema(ObserverDeciderSchema):
    harmony_priority: Literal["social_harmony_is_paramount", "personal_truth_over_harmony"]
    prompt_text: ClassVar[str] = (
        "Does the speaker prioritize **maintaining social harmony, appropriate emotional demeanor, and external pleasantries** (social_harmony_is_paramount), "
        "or do they prioritize **adherence to personal values and internal truth, even if it disrupts social comfort** (personal_truth_over_harmony)?\n"
        "Return { \"harmony_priority\":\"personal_truth_over_harmony\" }."
    )


class CoreSensitivitySchema(ObserverDeciderSchema):
    core_sensitivity: Literal["sensitive_to_being_called_stupid", "sensitive_to_being_unvalued"]
    prompt_text: ClassVar[str] = (
        "Is the speaker acutely sensitive to criticism related to **their logic, intelligence, or reasons** (sensitive_to_being_called_stupid), "
        "or to criticism related to **their intrinsic worth, values, or importance** (sensitive_to_being_unvalued)?\n"
        "Return { \"core_sensitivity\":\"sensitive_to_being_called_stupid\" }."
    )

class LogicConsistencySchema(ObserverDeciderSchema):
    consistency_focus: Literal[
        "intolerant_of_internal_inconsistencies",
        "tolerates_inconsistency_if_it_works",
    ]
    prompt_text: ClassVar[str] = (
        "Is the speaker highly sensitive to **small, internal, subjective logical inconsistencies** (intolerant_of_internal_inconsistencies), "
        "or do they prioritize **objective outcomes and system efficiency, tolerating logical flaws if the result works for the Tribe** (tolerates_inconsistency_if_it_works)?\n"
        "Return { \"consistency_focus\":\"intolerant_of_internal_inconsistencies\" }."
    )


class LogicHoningMethodSchema(ObserverDeciderSchema):
    honing_method: Literal["hones_reasons_in_internal_lab", "hones_reasons_in_external_action"]
    prompt_text: ClassVar[str] = (
        "Does the speaker primarily hone their logical structure and reasons through **internal, solitary analysis and contemplation** (hones_reasons_in_internal_lab), "
        "or through **external trial-and-error, action, and seeking feedback on efficiency** (hones_reasons_in_external_action)?\n"
        "Return { \"honing_method\":\"hones_reasons_in_internal_lab\" }."
    )


class ValueFlexibilitySchema(ObserverDeciderSchema):
    value_flexibility: Literal["unwilling_to_change_core_values", "willing_to_change_self_for_tribe"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's personal identity (Self) generally **unwilling to compromise its core values or beliefs** even when facing social rejection (unwilling_to_change_core_values), "
        "or is the speaker prone to **'morphing' or adjusting their persona** to gain external validation and acceptance (willing_to_change_self_for_tribe)?\n"
        "Return { \"value_flexibility\":\"unwilling_to_change_core_values\" }."
    )


class EmotionalHidingSchema(ObserverDeciderSchema):
    hiding_mechanism: Literal["hides_shame_through_isolation", "hides_shame_through_overcompensation"]
    prompt_text: ClassVar[str] = (
        "When dealing with personal shame or feelings of worthlessness, does the speaker tend to **withdraw and process privately** (hides_shame_through_isolation), "
        "or do they tend to **overcompensate by increasing external activity, socializing, or seeking intense Tribe validation** (hides_shame_through_overcompensation)?\n"
        "Return { \"hiding_mechanism\":\"hides_shame_through_isolation\" }."
    )


class DramaSourceSchema(ObserverDeciderSchema):
    drama_source: Literal["drama_from_logic_failures", "drama_from_value_conflicts"]
    prompt_text: ClassVar[str] = (
        "Is the speaker's recurring personal 'drama' rooted in **frustration with logical errors, system failures, or practical inefficiencies** (drama_from_logic_failures), "
        "or in **conflicts over values, feelings of being unappreciated, or social/emotional atmosphere problems** (drama_from_value_conflicts)?\n"
        "Return { \"drama_source\":\"drama_from_logic_failures\" }."
    )


class CriticalSensitivitySchema(ObserverDeciderSchema):
    critical_sensitivity: Literal["sensitive_to_logic_flaws", "sensitive_to_value_flaws"]
    prompt_text: ClassVar[str] = (
        "Is the speaker more profoundly hurt by criticism that attacks **their intelligence, competence, or logical reasoning** (sensitive_to_logic_flaws), "
        "or by criticism that attacks **their worth, values, or importance** (sensitive_to_value_flaws)?\n"
        "Return { \"critical_sensitivity\":\"sensitive_to_logic_flaws\" }."
    )
