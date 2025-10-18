#!/usr/bin/env bash

# Validate base-voice-id
if [[ -z "${GITHUB_EVENT_INPUTS_BASE_VOICE_ID}" ]]; then
  echo "❌ base-voice-id is required" >&2
  exit 1
fi
if ! [[ "${GITHUB_EVENT_INPUTS_BASE_VOICE_ID}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "❌ base-voice-id format invalid: ${GITHUB_EVENT_INPUTS_BASE_VOICE_ID}" >&2
  exit 1
fi

# Validate target-voice-id
if [[ -z "${GITHUB_EVENT_INPUTS_TARGET_VOICE_ID}" ]]; then
  echo "❌ target-voice-id is required" >&2
  exit 1
fi
if ! [[ "${GITHUB_EVENT_INPUTS_TARGET_VOICE_ID}" =~ ^[a-zA-Z0-9_-]+$ ]]; then
  echo "❌ target-voice-id format invalid: ${GITHUB_EVENT_INPUTS_TARGET_VOICE_ID}" >&2
  exit 1
fi

# Validate tts-provider
if [[ "${GITHUB_EVENT_INPUTS_TTS_PROVIDER}" != "edge" && "${GITHUB_EVENT_INPUTS_TTS_PROVIDER}" != "google" ]]; then
  echo "❌ tts-provider must be one of: edge, google" >&2
  exit 1
fi

# Validate num-words
if ! [[ "${GITHUB_EVENT_INPUTS_NUM_WORDS}" =~ ^[0-9]+$ ]] || [[ "${GITHUB_EVENT_INPUTS_NUM_WORDS}" -lt 1 ]]; then
  echo "❌ num-words must be a positive integer" >&2
  exit 1
fi

# Validate textgen-provider
if [[ "${GITHUB_EVENT_INPUTS_TEXTGEN_PROVIDER}" != "chatgpt" && "${GITHUB_EVENT_INPUTS_TEXTGEN_PROVIDER}" != "deepseek" && "${GITHUB_EVENT_INPUTS_TEXTGEN_PROVIDER}" != "perplexity" ]]; then
  echo "❌ textgen-provider must be one of: chatgpt, deepseek, perplexity" >&2
  exit 1
fi

echo "All workflow inputs are valid."
echo "valid=true" >> "$GITHUB_OUTPUT"
