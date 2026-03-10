# Evaluation Report — Intelligent Issue Detection Service

**Date:** 2026-03-10
**Environment:** Apple M1 Pro, Ollama qwen2.5:7b, 155 issues indexed (150 local + 5 Linear)

## Test Results Summary

| # | Query | Type | Classification | Duplicates Found | Time |
|---|-------|------|----------------|-----------------|------|
| 1 | APAC login/session failures | Bug (has duplicates) | ✅ is_issue=True (0.90) | ✅ 3 found (0.96, 0.85, 0.81) | 259s |
| 2 | Dark mode not saving preference | Bug (no duplicates) | ✅ is_issue=True (0.95) | ✅ 0 (correct) | 286s |
| 3 | Sprint retrospective meeting | Non-issue | ✅ is_issue=False (0.80) | ✅ 0 (correct) | 76s |
| 4 | Credit card Invalid Merchant ID | Bug (has duplicates) | ✅ is_issue=True (0.90) | ✅ 5 found (0.98, 0.97, 0.96, 0.96, 0.96) | 304s |
| 5 | 2FA feature request | Non-issue | ✅ is_issue=False (0.80) | ✅ 0 (correct) | 89s |

**Overall: 5/5 correct classifications, 5/5 correct duplicate behavior**

---

## Detailed Results

### Query 1 — Login/Session Issue (Expect Duplicates)

**Input:** "Several customers in the APAC region are unable to log into their accounts even though they are entering the right password. The password reset option is also broken and just shows a vague error."

**Result:**
- Classification: ✅ Bug, HIGH priority, confidence 0.90
- Extracted title: "APAC Users Can't Log In or Reset Password"
- Generated description: 281 chars, plain-text format matching database style
- Duplicates: 3 found
  - "User unable to login with valid credentials" (score: 0.96) — shares identical description about login failure + broken password reset
  - "Biometric authentication failing" (score: 0.85) — shares identical description
  - "Magic link authentication not working" (score: 0.81) — shares identical description

**Assessment:** ✅ Correctly identified cross-title duplicates — these three issues have completely different titles but the same underlying description about login/session problems. This is exactly the scenario the retrieve+rerank approach was designed to catch.

### Query 2 — Dark Mode Bug (No Duplicates Expected)

**Input:** "The dark mode toggle on the settings page is not saving the preference. Every time I refresh the page it reverts back to light mode."

**Result:**
- Classification: ✅ Bug, MEDIUM priority, confidence 0.95
- Extracted title: "Dark mode toggle not saving preference"
- Category: ui/ux
- Duplicates: 0 found

**Assessment:** ✅ Correctly classified as a UI bug. No duplicates exist in the dataset for this issue — the pipeline correctly returned an empty list rather than false positives.

### Query 3 — Meeting Request (Non-Issue)

**Input:** "Can we schedule a sprint retrospective for Friday at 3pm? We need to discuss the Q1 roadmap and assign new tasks to the frontend team."

**Result:**
- Classification: ✅ is_issue=False, confidence 0.80
- Reason: "This is a meeting request and does not describe a technical problem."
- Pipeline exited early after classification (steps 2-4 skipped)

**Assessment:** ✅ Correctly rejected. The pipeline correctly identified this as a scheduling request and did not waste LLM calls on extraction/generation.

### Query 4 — Payment Processing Issue (Expect Duplicates)

**Input:** "Credit card payments are failing randomly with an Invalid Merchant ID error code. It seems to happen about 15 percent of the time during certain hours."

**Result:**
- Classification: ✅ Bug (integration), MEDIUM priority, confidence 0.90
- Duplicates: 5 found (max returned)
  - "Contactless payment failing for merchant" (score: 0.98)
  - "Cache invalidation matter for key" (score: 0.97)
  - "Cache invalidation matter for key" (score: 0.96) — different issue ID, same title
  - "Duplicate payment detected for user" (score: 0.96)
  - "Failed ACH transfer for account" (score: 0.96)

**Assessment:** ✅ All 5 duplicates share the same underlying description about credit card authorization failing with 'Invalid Merchant ID'. The cross-title detection works — "Cache invalidation matter", "Contactless payment failing", and "Failed ACH transfer" are completely different titles but all describe the same payment processor integration problem.

### Query 5 — Feature Request (Edge Case)

**Input:** "We need to add two-factor authentication support for all user accounts. Currently there is no 2FA option and several enterprise clients have flagged this as a security gap."

**Result:**
- Classification: ✅ is_issue=False, confidence 0.80
- Reason: "This is a feature request for adding 2FA, not a current technical problem."

**Assessment:** ✅ Correctly distinguished a feature request from a bug report. The classifier recognized that the absence of 2FA is not a malfunction but a missing capability.

---

## Pipeline Performance Breakdown

| Stage | Query 1 | Query 2 | Query 4 | Avg (issues) |
|-------|---------|---------|---------|--------------|
| **Classify** (Ollama) | 57.1s | 59.9s | 30.6s | 49.2s |
| **Extract** (Ollama) | 126.4s | 136.8s | 176.1s | 146.4s |
| **Generate** (Ollama) | 74.4s | 88.2s | 95.9s | 86.2s |
| **Retrieve** (ChromaDB) | 1.3s | 0.6s | 1.2s | 1.0s |
| **Rerank** (CrossEncoder) | 0.2s | 0.4s | 0.5s | 0.4s |
| **Total** | 259s | 286s | 304s | 283s |

**Key observations:**
- LLM calls (classify + extract + generate) account for ~99.5% of total time
- Extraction is the slowest stage (~146s avg) — Ollama function calling with structured output
- Retrieve + Rerank together take ~1.4s — negligible compared to LLM stages
- Non-issue queries complete in ~76-89s (classify-only, 4x faster)

---

## Conclusion

The pipeline works correctly across all 5 test cases:

1. **Classification accuracy:** 5/5 — correctly separates bugs from non-issues and feature requests
2. **Duplicate detection precision:** No false positives observed — empty results when no duplicates exist
3. **Duplicate detection recall:** Cross-title duplicates detected with high confidence (0.81-0.98) across both test cases with known duplicates
4. **Retrieve + Rerank performance:** Sub-2-second duplicate search across 155 issues, independent of LLM latency
5. **Error handling:** Structured JSON error responses on failure (tested separately)

The main bottleneck is LLM inference time on the local qwen2.5:7b model (~283s per issue query). This is expected for a 7B parameter model running on CPU/GPU locally without quantization optimization.
