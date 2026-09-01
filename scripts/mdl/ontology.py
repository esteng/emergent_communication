"""The fixed SaveVeyru referent space.

Taken verbatim from the agent system prompts, not inferred from what the agents said.
The emergent codes ground out into exactly these 37 non-numeric referents:
14 failure motifs, 14 procedure templates, 6 faces and 3 intensity levels. Durations
are numeric and so are not part of the count.
"""

# ── motifs: name -> canonical symptom description (engineer brief) ───────────────
MOTIFS = {
    "Alignment Collapse": "Faces are flickering randomly between light and dark patches that form no pattern. The hum is broken and irregular, starting and stopping with no rhythm. Edges look normal but the face surfaces keep shifting chaotically.",
    "Drift Escalation": "The light on each face keeps sliding slowly across the surface like colors drifting. Edges look slightly blurred, as if the boundaries between faces are smearing. The hum is wavering up and down in pitch without settling.",
    "Echo Saturation": "It is much too bright, almost hard to look at. The hum has a layered quality, like multiple tones stacked on top of each other. Some faces show frozen patterns that do not change or respond when touched.",
    "Leak Instability": "The corners are noticeably dimmer than the rest, almost dark. Several edges look faint like they are fading out. The center of each face is fine but the perimeter is losing light. The hum sounds thin and hollow at the edges.",
    "Low Intensity": "It is dim overall, all faces are faint. The hum is barely audible, more of a whisper. Patterns on the faces are visible but washed out, like the whole thing is running low.",
    "High Intensity": "All faces are blazing with painfully bright white light. The hum is a loud harsh buzz that vibrates the surface it sits on. There is noticeable heat radiating from the faces.",
    "Phase Inversion": "Opposite faces alternate between bright and dark in a strict pulsing rhythm. The hum oscillates between two distinct tones in sync with the pulses. Edges flash in time with the face pulses.",
    "Resonance Cascade": "One face is dramatically brighter than the others with intense localized vibration. A high-pitched whine has replaced the normal hum near that face. The other faces appear normal by comparison.",
    "Corner Deadlock": "One or two corners glow intensely bright while the rest looks normal. A clicking or ticking sound replaces the hum near the bright corners. Heat is concentrated at the bright corners.",
    "Boundary Softening": "Edges appear to wobble or flex when touched. Faces look slightly curved or bulging, the box shape is subtly distorted. The hum sounds muffled as if underwater.",
    "Propagation Stall": "Light patterns have frozen completely — dim, not bright, and totally still. The hum has dropped to silence. The surface feels cold and does not respond to touch or tapping.",
    "Harmonic Split": "The hum has split into two or more competing tones that clash. Light patterns alternate between two different configurations. Edges shimmer as if two patterns are fighting for dominance.",
    "Thermal Bleed": "It is very hot all over but dim. The hum is a low rumble instead of a tone. Surfaces feel rough or gritty, and the light has a reddish tint.",
    "Core Void": "Faces glow normally at the surface but it sounds hollow when tapped. Holding it up to a light source shows a dark center — light does not penetrate. The hum sounds thin and surface-level, with no depth.",
}

# ── procedure templates: id -> canonical text (slots blanked) + short label ──────
# Slots removed so a concrete procedure string can be matched back to its template.
TEMPLATES = {
    "TONE_ALL":          "Sound a sustained tone near all six faces simultaneously for seconds, starting from the face. Let the tone fade naturally and wait for the hum to stabilize.",
    "BELL_ALT":          "Chime a bell near two opposite faces, starting from the face. Alternate the chime between the two faces seconds pause between chimes for five cycles at tone.",
    "CLOTH_ADJ_E":       "Drape a cloth over two adjacent edges near the face for seconds at coverage. Then chime a bell three times near the face.",
    "STONE_CORNERS":     "Warm each corner of the face by holding a heated stone nearby for seconds at warmth, in sequence. Then trace each edge of the face with a finger.",
    "STONE_BESIDE_ROT":  "Place a warm stone beside the face at warmth for seconds. Rotate and repeat for each face.",
    "FAN_OPP":           "Drape a cool cloth over the Veyru for seconds. Remove the cloth and fan cool air across the face and the opposite face for seconds.",
    "LAMP_DIM_BR":       "Illuminate the face with a dim lamp and the opposite face with a bright lamp simultaneously at brightness for seconds, without moving the lamps.",
    "CLOTH_FOLD_FACE":   "Drape a folded cloth over the face for seconds at coverage. Then chime a bell near each edge of the face twice.",
    "CORNER_BELL_STONE": "At each corner of the face, chime a bell briefly at tone, then warm the two edges meeting at that corner with a heated stone for seconds.",
    "BOARD_FACE_ROT":    "Rest a flat board against the face at contact for seconds. Rotate and repeat for each face-pair.",
    "BELL_CTR_STONE":    "Chime a bell near the center of each face once at tone, starting from the face. Then place a warm stone beside the Veyru for seconds.",
    "SOFT_REST":         "Place the Veyru on a soft surface with the face up at contact. Let it rest undisturbed for seconds.",
    "FAN_ALL":           "Drape a cool cloth over the Veyru for seconds. Remove and fan cool air across all six faces for seconds, starting from the face.",
    "ROTATE_TONE":       "Rotate the Veyru slowly over seconds while sounding a steady tone at volume near opposite faces, starting from the face. After rotation, chime a bell once near each corner.",
}

FACES = ["top", "bottom", "front", "back", "left", "right"]
INTENSITIES = ["gentle", "moderate", "firm"]
