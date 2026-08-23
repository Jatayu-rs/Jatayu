# Bhashini / ULCA Integration Notes

**Owner:** Person F · **Last updated:** 23 Aug 2026
**Status:** API flow and language support researched and documented. **Registration still to be done — see §1.**
**Scope note:** multilingual is a *future work* item, not part of the 4-week build. This document exists so that when we do it, nobody has to read the Bhashini docs from scratch.

> ⚠️ **Credentials do not go in this file.** The `userID` and API key belong in the team chat only. This repository is public — anything committed here is public forever, including in git history after a delete.

---

## Why Bhashini

It is the Government of India's national language AI platform (MeitY / National Language Translation Mission). Two reasons it's on our list:

1. **It does speech, not just text.** Translation, speech-to-text (ASR), text-to-speech (TTS) and transliteration. A user could *speak* a query about a satellite image and *hear* the answer.
2. **It is a national initiative**, which reads well to Indian evaluators in a way that a US commercial API does not.

---

## 1. Registration — ~10 minutes, must be done by a person

**I could not do this part.** It requires creating an account in a named person's name, accepting terms of service, and clicking a verification link in that person's inbox. That has to be you (or whoever will own the account). The steps below are exact — the whole thing is form-filling.

1. Go to **`https://bhashini.gov.in/ulca/user/register`**
2. Fill in the registration form and submit.
3. **Check email for the verification link and click it.** The docs specifically warn to *check the spam folder* — this mail is commonly filtered.
4. Log in at **`https://bhashini.gov.in/ulca/user/login`**
5. Go to **My Profile** (`https://bhashini.gov.in/ulca/profile`)
6. Click **Generate** to create an API key.
   - **App name must be lower-case words, underscores allowed** — e.g. `satquery_ai`. It rejects capitals and spaces.
   - **You get a maximum of 5 keys per account.** Don't burn them experimenting; one key is enough for the whole team. Keys can be revoked individually if one leaks.
7. From **My Profile**, copy the two values:
   - **`userID`**
   - **`ulcaApiKey`**
8. **Post both in the team chat. Do not commit them anywhere.** Then tick the box below.

- [ ] Account registered and email verified
- [ ] API key generated (app name: `satquery_ai`)
- [ ] `userID` and `ulcaApiKey` posted to team chat
- [ ] This line updated with who owns the account: **_(name)_**

*Eligibility note: the onboarding docs state no restriction to Indian entities or government use. If registration is refused for any reason, that's new information and worth flagging — I found nothing suggesting it should be.*

---

## 2. The API flow — numbered steps for a coder

Bhashini is a **two-call pattern**. You cannot go straight to translating. First you ask "which model can do this job?" and get an identifier back; then you use that identifier to do the actual work.

```
  YOU                    meity-auth.ulcacontrib.org         dhruva-api.bhashini.gov.in
   │                                │                                  │
   │ ①  config call ───────────────►│                                  │
   │    (userID + ulcaApiKey)       │                                  │
   │                                │                                  │
   │ ◄─── serviceId ────────────────│                                  │
   │      callbackUrl               │                                  │
   │      inference Authorization   │                                  │
   │                                                                   │
   │ ②  compute call ─────────────────────────────────────────────────►│
   │    (serviceId + inference Authorization + your text)              │
   │                                                                   │
   │ ◄─── translated text / audio ─────────────────────────────────────│
```

### ⚠️ The one thing everyone gets wrong

**There are two different credentials and they are not interchangeable.**

| | Used in | Where it comes from |
|---|---|---|
| `userID` + `ulcaApiKey` | **Call ①** only | Your ULCA profile page |
| `Authorization` header | **Call ②** only | The *response body* of call ① |

Using your `ulcaApiKey` on call ② will fail with an auth error. This is the single most common Bhashini integration bug.

---

### Step 0 — Prerequisites

You need:
- `userID` and `ulcaApiKey` from §1 (get them from team chat)
- A **`pipelineId`**. Two are published and you can hardcode one:

| Provider | pipelineId |
|---|---|
| **MeitY** | `64392f96daac500b55c543cd` |
| **AI4Bharat** | `643930aa521a4b1ba0f4c41d` |

> Start with the **MeitY** pipeline — it's the one used in most working examples. ⚠️ Other values for the AI4Bharat ID circulate on blogs and in sample code; the one above is from the official docs. If it 404s, check the [Pipeline Search Call](https://bhashini.gitbook.io/bhashini-apis/pipeline-search-call) page rather than trusting a blog.

---

### Step 1 — The config call ("which model can do this?")

```
POST https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline
```

**Headers** — note the exact casing, and **no `Bearer` prefix**:

```
userID:     <your userID>
ulcaApiKey: <your ulcaApiKey>
Content-Type: application/json
```

**Body** — declare the task(s) you want and the languages:

```json
{
  "pipelineTasks": [
    {
      "taskType": "translation",
      "config": {
        "language": {
          "sourceLanguage": "en",
          "targetLanguage": "hi"
        }
      }
    }
  ],
  "pipelineRequestConfig": {
    "pipelineId": "64392f96daac500b55c543cd"
  }
}
```

- `taskType` is one of: **`translation`**, **`asr`**, **`tts`**, **`transliteration`**
- For `asr` and `tts` you supply only `sourceLanguage`, not `targetLanguage`
- You can chain tasks — put `asr`, then `translation`, then `tts` in the array to build a full speak-in/speak-out pipeline in one go

---

### Step 2 — Pull three things out of the response

The response looks like this (trimmed):

```json
{
  "languages": [
    { "sourceLanguage": "en", "targetLanguageList": ["hi", "bn", "ta", "..."] }
  ],
  "pipelineResponseConfig": [
    {
      "taskType": "translation",
      "config": [
        {
          "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4",
          "modelId": "641d1d6a8ecee6735a1b372a",
          "language": { "sourceLanguage": "en", "targetLanguage": "hi" }
        }
      ]
    }
  ],
  "pipelineInferenceAPIEndPoint": {
    "callbackUrl": "https://dhruva-api.bhashini.gov.in/services/inference/pipeline",
    "inferenceApiKey": {
      "name": "Authorization",
      "value": "cZVqccgm-LTAzxQVp6jjznmSR5RgKM"
    },
    "isMultilingualEnabled": true,
    "isSyncApi": true
  }
}
```

**Extract exactly these three:**

| # | What | JSON path |
|---|---|---|
| 1 | **serviceId** | `pipelineResponseConfig[0].config[0].serviceId` |
| 2 | **callbackUrl** | `pipelineInferenceAPIEndPoint.callbackUrl` |
| 3 | **auth header name + value** | `pipelineInferenceAPIEndPoint.inferenceApiKey.name` and `.value` |

If you requested multiple tasks, `pipelineResponseConfig` is an array **in the order you asked for them** — index `[0]` is your first task, `[1]` the second, and so on. Match on `taskType` rather than trusting position if you want to be safe.

> **Bonus:** the `languages` array in this response is the **authoritative, live list of what's actually supported**. One config call with just a `pipelineId` and you can generate the language tables in §3 yourself, current as of today. Do this rather than trusting my tables or any blog post.

---

### Step 3 — The compute call (the actual work)

```
POST <the callbackUrl from step 2>
```
— in practice `https://dhruva-api.bhashini.gov.in/services/inference/pipeline`

**Headers:**

```
Authorization: <the inferenceApiKey.value from step 2>
Content-Type: application/json
```

**Body** — same shape as the config call, but now with `serviceId` filled in and your actual input attached:

```json
{
  "pipelineTasks": [
    {
      "taskType": "translation",
      "config": {
        "language": {
          "sourceLanguage": "en",
          "targetLanguage": "hi"
        },
        "serviceId": "ai4bharat/indictrans-v2-all-gpu--t4"
      }
    }
  ],
  "inputData": {
    "input": [
      { "source": "Show me how forest cover changed near the Sundarbans since 2015." }
    ],
    "audio": [
      { "audioContent": null }
    ]
  }
}
```

The translated text comes back under `pipelineResponse[0].output[0].target`.

---

### Task-specific body differences

**ASR (speech → text)** — audio goes in, `source` is null:

```json
{
  "pipelineTasks": [{
    "taskType": "asr",
    "config": {
      "language": { "sourceLanguage": "hi" },
      "serviceId": "<from config call>",
      "audioFormat": "wav",
      "samplingRate": 16000
    }
  }],
  "inputData": {
    "input": [{ "source": null }],
    "audio": [{ "audioContent": "<base64-encoded wav>" }]
  }
}
```

**TTS (text → speech)** — adds `gender`, returns base64 audio:

```json
{
  "pipelineTasks": [{
    "taskType": "tts",
    "config": {
      "language": { "sourceLanguage": "hi" },
      "serviceId": "<from config call>",
      "gender": "female"
    }
  }],
  "inputData": {
    "input": [{ "source": "यह उत्तर है।" }],
    "audio": [{ "audioContent": null }]
  }
}
```

---

### Gotchas, collected

1. **Two credentials, two calls.** See the warning above. This will be your first bug.
2. **The `audio` block is required even for translation** — pass `[{"audioContent": null}]`. Omitting it can 400.
3. **`source` is required even for ASR** — pass `null`.
4. **ASR audio must be base64-encoded**, `wav`, 16000 Hz sampling rate.
5. **The compute call goes to `dhruva-api...`, not `meity-auth...`.** Different host from call ①.
6. **Cache the config response** — it's slow and unnecessary to call before every translation. But don't hardcode `serviceId` permanently; providers get rotated. Cache per session, refresh on auth failure.
7. **Language codes are ISO-639** and mixed-length: two letters for most (`hi`, `bn`, `ta`), three for several (`brx` Bodo, `kok` Konkani, `mai` Maithili, `mni` Manipuri, `sat` Santali, `doi` Dogri).
8. **5 API keys maximum per account.** Share one key across the team; don't generate per-person.

---

## 3. Language support — the three lists are NOT the same

This is the part that matters for planning. Translation covers far more languages than speech does, so a language being "supported by Bhashini" does **not** mean you can build a voice interface in it.

**Supported by all three (translation + ASR + TTS) — safe for a full voice round-trip:**

Assamese `as` · Bengali `bn` · Bodo `brx` · Dogri `doi` · English `en` · Gujarati `gu` · Hindi `hi` · Kannada `kn` · Konkani `kok` · Maithili `mai` · Malayalam `ml` · Manipuri `mni` · Marathi `mr` · Nepali `ne` · Odia `or` · Punjabi `pa` · Sanskrit `sa` · Santali `sat` · Sindhi `sd` · Tamil `ta` · Telugu `te` · Urdu `ur` · Bhojpuri `bho`

**≈23 languages.** That is the realistic ceiling for a speech-driven demo.

### Where the lists diverge

| Language | Translation | ASR | TTS | Note |
|---|:---:|:---:|:---:|---|
| Kashmiri `ks` | ✅ | ✅ | ❌ | **Can hear it, can't speak it.** The notable gap. |
| Chhattisgarhi `hne` | ❌ | ❌ | ✅ | TTS only |
| Rajasthani `raj` | ❌ | ❌ | ✅ | TTS only |
| Magahi `mag` | ✅ | ❌ | ✅ | No speech input |
| Sinhala `si` | ✅ | ❌ | ❌ | Text only |
| Awadhi, Braj, Gondi, Ho, Khasi, Mizo, Tulu, Kangri, Hinglish | ✅ | ❌ | ❌ | **Translation only — no speech at all** |

**Model counts, per the official docs:** translation 8 models · ASR 10 models · TTS 5 models · transliteration 1 model (IndicXlit).

**Transliteration** (script conversion, not translation) covers roughly the translation list minus the rarest languages — useful if we ever want to render a Hindi place name in Latin script.

> ⚠️ **Two caveats on these tables.**
> 1. **The ISO codes are mine**, mapped from the language names in the Bhashini docs. The docs list names, not codes. Confirm codes against the `languages` array in a real config-call response before relying on them.
> 2. **Hindi did not appear in the translation list** as published on the docs page I read. That is almost certainly a documentation/extraction artifact — Hindi is Bhashini's flagship language and every working code sample translates to it. I've included it above, but flagging that I inferred it rather than read it.
>
> Both caveats disappear the moment someone runs one config call. **Do that before we put a language count on a slide.**

---

## 4. If we use this in the pitch

Honest framing for a judge:

> "Multilingual query support is designed, not built. We've mapped the Bhashini integration — it's a two-call pipeline against the MeitY endpoint, and roughly 23 Indian languages support a full speech-in/speech-out round trip. We scoped it out of the four-week build deliberately rather than ship it half-working."

That is a stronger answer than a broken voice demo, and "we chose not to build this yet, and here's the integration spec" reads as judgement rather than a gap.

---

## Sources

- [Bhashini API docs — overview](https://bhashini.gitbook.io/bhashini-apis) · [pre-requisites and onboarding](https://bhashini.gitbook.io/bhashini-apis/pre-requisites-and-onboarding)
- [Pipeline config call — request payload](https://bhashini.gitbook.io/bhashini-apis/pipeline-config-call/request-payload) · [response payload](https://bhashini.gitbook.io/bhashini-apis/pipeline-config-call/response-payload)
- [Pipeline compute call — request payload](https://bhashini.gitbook.io/bhashini-apis/pipeline-compute-call/request-payload)
- [Pipeline search call (pipeline IDs)](https://bhashini.gitbook.io/bhashini-apis/pipeline-search-call)
- [Available models and languages per task](https://dibd-bhashini.gitbook.io/bhashini-apis/available-models-for-usage)
- [ULCA registration](https://bhashini.gov.in/ulca/user/register) · [login](https://bhashini.gov.in/ulca/user/login) · [profile](https://bhashini.gov.in/ulca/profile)
- [Working reference implementation (Python)](https://github.com/AdityaKukreti/bhashini-api/blob/main/BhashiniAPI.py) · [bhashini_translator package](https://github.com/dteklavya/bhashini_translator)
