#!/usr/bin/env python3
"""
================================================================================
 AI & TECH NEWS -> SHORT-FORM VIDEO GENERATOR  (v4, single-file, production-grade)
================================================================================

PIPELINE (DAILY BATCH MODEL):
  1. Gather candidates: yesterday's leftover queue (queued_news.txt) + fresh
     whitelisted, non-duplicate headlines from Google News RSS (global/US tech).
  2. ONE Gemini call ranks all candidates by viral potential.
  3. Take the top DAILY_STORY_TARGET (3) stories. For each: fact-check + write
     an English viral script with a duration-enforcement retry loop (30-40s).
  4. edge-tts -> natural US English voiceover (with word-level timestamps).
  5. Pexels -> 1080x1920 (9:16) stock footage, cut every 2-3s, dynamic PIL captions.
  6. Upload to YouTube as `private` with `publishAt` set to the next available
     one of 3 fixed US peak time slots (EST/EDT) - it goes public automatically
     at that time, no manual step needed.
  7. Deliver the same video + copy-paste metadata to Telegram as a review copy.
  8. Any candidates not selected today are written back to queued_news.txt so
     no good story is wasted - they're re-ranked alongside fresh news tomorrow.

--------------------------------------------------------------------------------
CHANGELOG vs v3 (Hindi single-story hourly pipeline -> English daily-batch pipeline)
--------------------------------------------------------------------------------
  [CHANGED] TTS: hi-IN-MadhurNeural -> en-US-AndrewNeural (en-US-AriaNeural as
            the documented alternative).
  [REMOVED] Devanagari font verification/download - replaced with
            find_caption_font(), which uses standard DejaVu/Liberation/Arial
            fonts already present on most systems, with a Roboto-Bold web
            fallback. No apt-get font package needed for English captions.
  [CHANGED] NEWS_RSS_URL -> US/global English edition; SOURCE_WHITELIST ->
            TechCrunch, The Verge, Ars Technica, Wired, Engadget, VentureBeat.
  [CHANGED] GEMINI_PROMPT_TEMPLATE rewritten for an English viral tech
            scriptwriter persona (hook / fast-paced suspense body / like+
            subscribe CTA), JSON key `script_hi` -> `script_en`.
  [ADD] Daily queue + smart ranking: scan_fresh_candidates() + load_queue()
        feed a single score_candidate_stories() Gemini call; top 3 get
        produced, the rest are persisted to queued_news.txt for future days.
  [ADD] YouTube Data API v3 scheduled upload (upload_video_to_youtube) mapped
        to 3 fixed daily US peak slots via compute_slot_datetime(), timezone-
        aware (America/New_York, DST-safe) using zoneinfo.
  [KEPT] The v3 fixes: mark-processed-after-primary-delivery ordering (now
         after local render, since YouTube+Telegram are both best-effort
         delivery channels), fuzzy dedup, Gemini call budget + pacing per
         story, RSS structure debug dump.
  [ADD]  Slot-time jitter: compute_slot_datetime() now picks a random minute
         inside each slot's window (e.g. 8:00-10:00) instead of a fixed
         HH:MM, so scheduled publish times vary day to day.
  [ADD]  YouTube-only retry queue (pending_youtube_uploads.txt): a story
         whose video rendered fine but whose YouTube upload failed (bad
         OAuth token, quota, transient API error) is never re-rendered from
         scratch - only the upload is retried, using the already-saved mp4,
         at the start of every subsequent daily batch, up to
         MAX_YOUTUBE_UPLOAD_RETRIES attempts before being dropped.
  [ADD]  Automatic thumbnail generation: grabs a frame from the rendered
         video and overlays a bold headline (same zero-cost PIL pipeline as
         the captions), then uploads it via youtube.thumbnails().set(). NOTE:
         YouTube's Shorts swipe feed usually ignores custom thumbnails and
         auto-picks its own frame regardless - this is a platform limitation
         reflected honestly in generate_thumbnail()'s docstring, not a bug
         here. The thumbnail still applies to search results, the channel's
         video grid, and playlists/Watch Later.
  [ADD]  _ensure_shorts_hashtag(): guarantees '#Shorts' is present in the
         hashtag list (and therefore the description) even if Gemini forgets
         it, as an extra signal toward YouTube's Shorts shelf classification.
  [ADD]  Production-polish pass: background music mixed under the voice
         (drop your own royalty-free tracks in MUSIC_DIR - none bundled, so
         licensing is always in your control); crossfade transitions +
         subtle Ken Burns zoom between background segments instead of hard
         cuts; consistent color grading (apply_color_grade) for a repeatable
         channel "look"; optional logo watermark + branded intro sting (drop
         a transparent PNG at LOGO_PATH - skipped gracefully if absent);
         captions upgraded from static per-chunk blocks to karaoke-style
         word-by-word highlighting (build_caption_clips now renders one
         image per WORD, reusing the same line layout, only the spoken
         word's color changes).

--------------------------------------------------------------------------------
ENVIRONMENT SETUP
--------------------------------------------------------------------------------
!apt-get update && apt-get install -y ffmpeg
!pip install -q feedparser google-generativeai edge-tts moviepy pillow numpy requests \
    google-api-python-client google-auth-oauthlib google-auth-httplib2

--------------------------------------------------------------------------------
YOUTUBE OAUTH SETUP (one-time, required before the first upload)
--------------------------------------------------------------------------------
RECOMMENDED for CI (GitHub Actions) - OAuth Playground, no local script run:
1. In Google Cloud Console, create an OAuth Client ID of type "Web application"
   for a project with the YouTube Data API v3 enabled. Add
   `https://developers.google.com/oauthplayground` as an Authorized redirect URI.
   Note the Client ID and Client Secret.
2. Go to https://developers.google.com/oauthplayground, click the gear icon
   (top right) and check "Use your own OAuth credentials" - paste your Client
   ID and Client Secret there.
3. In the left panel, find "YouTube Data API v3" and select the
   `https://www.googleapis.com/auth/youtube.upload` scope. Click "Authorize
   APIs", sign in with the channel's Google account, allow access.
4. Click "Exchange authorization code for tokens" - copy the Refresh Token.
5. Set three env vars / GitHub Secrets: YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET,
   YOUTUBE_REFRESH_TOKEN. That's it - no token file, no local script run needed.

ALTERNATIVE for local/Colab use - browser consent flow:
1. Create an OAuth Client ID of type "Desktop app" instead, download it as
   `client_secrets.json` next to this script.
2. Run this script ONCE on a machine with a browser (it opens a consent
   screen via `run_local_server`), which creates `youtube_token.json`. Copy
   that token file alongside the script wherever you run it afterward - it
   auto-refreshes. (This path is NOT used if the three env vars above are set.)

RUN:
    python news_to_video_v4.py

Only the CONFIGURATION block below needs editing.
================================================================================
"""

import os
import re
import sys
import json
import time
import html
import random
import difflib
import asyncio
import logging
import tempfile
import textwrap
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple, Set

import requests
import feedparser
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import edge_tts
import google.generativeai as genai

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    ImageClip,
    concatenate_videoclips,
    concatenate_audioclips,
    vfx,
)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # Python < 3.9
    from backports.zoneinfo import ZoneInfo  # pip install backports.zoneinfo

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request as GoogleAuthRequest
from googleapiclient.discovery import build as build_google_service
from googleapiclient.http import MediaFileUpload

# ==============================================================================
# 1. CONFIGURATION  --  REPLACE THESE VALUES
# ==============================================================================
# Every value below can be overridden by an environment variable of the same
# name (falls back to the hardcoded placeholder if the env var isn't set).
# This lets the exact same file run unmodified locally/Colab (edit the
# strings directly) or in CI such as GitHub Actions (set repo secrets as env
# vars - nothing in this file needs to change or be patched at deploy time).
GEMINI_API_KEY       = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY")
PEXELS_API_KEY        = os.environ.get("PEXELS_API_KEY", "YOUR_PEXELS_API_KEY")
TELEGRAM_BOT_TOKEN    = os.environ.get("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID      = os.environ.get("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")

YOUTUBE_CLIENT_SECRETS_FILE = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
YOUTUBE_TOKEN_FILE          = os.environ.get("YOUTUBE_TOKEN_FILE", "youtube_token.json")

# Preferred auth method for CI (GitHub Actions): a refresh token obtained
# once via Google's OAuth Playground (https://developers.google.com/oauthplayground)
# - no local browser flow or token file needed at all. If all three of these
# are set, get_youtube_service() uses them directly and skips the file-based
# flow below entirely.
YOUTUBE_CLIENT_ID           = os.environ.get("YOUTUBE_CLIENT_ID", "")
YOUTUBE_CLIENT_SECRET       = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
YOUTUBE_REFRESH_TOKEN       = os.environ.get("YOUTUBE_REFRESH_TOKEN", "")

# In CI (e.g. GitHub Actions cron), set this to 0 via the AUTO_LOOP_INTERVAL_HOURS
# env var - the scheduler triggers a fresh run, so the script should do ONE
# batch and exit, not sleep in a loop (which would just get killed at the
# job timeout and waste runner minutes for nothing).
AUTO_LOOP_INTERVAL_HOURS = int(os.environ.get("AUTO_LOOP_INTERVAL_HOURS", "24"))
PROCESSED_NEWS_FILE      = "processed_news.txt"
QUEUED_NEWS_FILE         = "queued_news.txt"
PENDING_YOUTUBE_UPLOADS_FILE = "pending_youtube_uploads.txt"   # stories rendered OK but whose YouTube upload failed

# ------------------------------------------------------------------------------
# Tuning knobs (safe to leave as-is)
# ------------------------------------------------------------------------------
NEWS_RSS_URL               = "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en"
GEMINI_MODEL                = "gemini-1.5-flash"
TTS_VOICE                   = "en-US-AndrewNeural"    # or "en-US-AriaNeural"
TTS_RATE                    = "+8%"
VIDEO_WIDTH                 = 1080
VIDEO_HEIGHT                 = 1920
CLIP_SEGMENT_MIN_SEC        = 2.0
CLIP_SEGMENT_MAX_SEC        = 3.0
PEXELS_RESULTS_PER_QUERY    = 5
MAX_BACKGROUND_CLIPS        = 10
WORDS_PER_CAPTION_CHUNK     = 4
OUTPUT_DIR                  = os.path.join(os.getcwd(), "output")
TEMP_DIR                    = tempfile.mkdtemp(prefix="n2v_")
MAX_GEMINI_RETRIES          = 3        # malformed-JSON / API-error retries, per single generate_script_package() call
MAX_DURATION_RETRIES        = 3        # outer duration-enforcement rewrite attempts, per story
MAX_GEMINI_CALLS_PER_CYCLE  = 6        # hard cap on Gemini calls for ONE story's script generation
GEMINI_MIN_INTERVAL_SEC     = 4.5      # min seconds between consecutive Gemini calls (free-tier RPM safety margin)
MAX_FEED_ENTRIES_SCANNED    = 40       # how many RSS entries to scan when building the candidate pool
REQUEST_TIMEOUT             = 60
TARGET_MIN_SEC               = 30.0
TARGET_MAX_SEC               = 40.0
FUZZY_DEDUP_THRESHOLD         = 0.82
FUZZY_DEDUP_MAX_COMPARE       = 500
DEBUG_RSS_STRUCTURE           = True

DAILY_STORY_TARGET          = 3        # videos produced per daily batch
MAX_CANDIDATES_TO_SCORE     = 15       # cap on candidates sent to the single ranking Gemini call
MAX_DAILY_ATTEMPTS          = 6        # safety cap on total stories attempted in one batch (incl. fact-check skips)

YOUTUBE_TIME_ZONE            = "America/New_York"     # EST/EDT, DST-safe via zoneinfo
# Each slot is a (start_hour, start_minute, end_hour, end_minute) window in local
# time; compute_slot_datetime() picks a random minute inside the window each time,
# so scheduled times vary day to day instead of always landing on the dot.
SLOT_WINDOWS = [
    (8, 0, 10, 0),    # Slot 1: Morning Rush
    (12, 0, 14, 0),   # Slot 2: Lunch Break
    (17, 0, 20, 0),   # Slot 3: Prime Time Evening
]
SLOT_MIN_LEAD_MINUTES        = 20      # don't schedule inside a window that ends within this many minutes (rolls to next day)
YOUTUBE_CATEGORY_ID          = "28"    # Science & Technology

# ------------------------------------------------------------------------------
# Polish: background music, transitions, and channel branding (all optional -
# every one of these degrades gracefully to "skip it" if the asset is missing,
# so the pipeline never breaks because a music folder or logo isn't set up yet)
# ------------------------------------------------------------------------------
MUSIC_DIR                    = os.environ.get("MUSIC_DIR", "music")   # put your own royalty-free .mp3/.wav tracks here
MUSIC_VOLUME                 = 0.12    # relative level under the voice track (0-1)
CROSSFADE_DURATION           = 0.35    # seconds of crossfade between background clips
ENABLE_KEN_BURNS             = True    # subtle continuous zoom-in per background segment
KEN_BURNS_ZOOM_AMOUNT        = 0.06    # max zoom over a segment's duration (0.06 = 6%)
COLOR_GRADE_SATURATION       = 1.08    # >1.0 = more saturated; consistent "look" across videos
COLOR_GRADE_CONTRAST         = 6       # moviepy vfx.lum_contrast contrast parameter
LOGO_PATH                    = os.environ.get("LOGO_PATH", os.path.join("assets", "logo.png"))  # transparent PNG, put your own here
WATERMARK_WIDTH_RATIO        = 0.16    # logo width as a fraction of video width
WATERMARK_OPACITY            = 0.55
INTRO_STING_DURATION         = 0.8     # seconds; added ON TOP of the 30-40s main content
MAX_YOUTUBE_UPLOAD_RETRIES   = 5       # cap on daily retry attempts for a failed upload before it's dropped from the pending queue

# Authoritative source whitelist: domain -> list of name fragments as they may
# appear in the RSS <source> tag or in the "Headline - Source Name" suffix.
SOURCE_WHITELIST: Dict[str, List[str]] = {
    "techcrunch.com": ["techcrunch"],
    "theverge.com": ["the verge", "verge"],
    "arstechnica.com": ["ars technica"],
    "wired.com": ["wired"],
    "engadget.com": ["engadget"],
    "venturebeat.com": ["venturebeat"],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("news2video")

_rss_debug_dumped = False
_last_gemini_call_ts = 0.0


# ==============================================================================
# 2. DATA MODELS
# ==============================================================================
@dataclass
class NewsItem:
    title: str
    summary: str
    link: str
    source_name: str


@dataclass
class ScriptPackage:
    script_en: str
    title: str
    description: str
    hashtags: List[str]
    tags: List[str]
    thumbnail_idea: str
    visual_keywords: List[str] = field(default_factory=list)
    is_credible: bool = True
    credibility_note: str = ""


@dataclass
class WordTiming:
    text: str
    start: float
    end: float


@dataclass
class CaptionChunk:
    text: str
    start: float
    end: float


class CredibilitySkipError(Exception):
    def __init__(self, note: str):
        super().__init__(note)
        self.note = note


class GeminiCallBudget:
    def __init__(self, max_calls: int):
        self.max_calls = max_calls
        self.used = 0

    def consume(self) -> bool:
        if self.used >= self.max_calls:
            return False
        self.used += 1
        return True

    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)


# ==============================================================================
# 3. CONFIG VALIDATION
# ==============================================================================
def validate_config() -> None:
    placeholders = {
        "GEMINI_API_KEY": GEMINI_API_KEY,
        "PEXELS_API_KEY": PEXELS_API_KEY,
        "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
        "TELEGRAM_CHAT_ID": TELEGRAM_CHAT_ID,
    }
    missing = [k for k, v in placeholders.items() if not v or v.startswith("YOUR_")]
    if missing:
        raise ValueError(
            "Config error - replace these at the top of the script before running: "
            + ", ".join(missing)
        )
    if not (YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN) and not os.path.exists(YOUTUBE_CLIENT_SECRETS_FILE):
        log.warning(
            "  No YouTube credentials found (neither YOUTUBE_CLIENT_ID/SECRET/REFRESH_TOKEN env vars "
            f"nor a '{YOUTUBE_CLIENT_SECRETS_FILE}' file) - YouTube uploads will be skipped "
            "(video still renders and goes to Telegram). See the OAuth setup notes at the top of this file."
        )


# ==============================================================================
# 4. NEWS: WHITELIST FILTER, DEDUP, CANDIDATE SCAN, QUEUE PERSISTENCE
# ==============================================================================
def _strip_html(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw or "")
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_headline(title: str) -> str:
    t = title.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def load_processed_headlines(path: str = PROCESSED_NEWS_FILE) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [_normalize_headline(line) for line in f if line.strip()]


def mark_headline_processed(title: str, path: str = PROCESSED_NEWS_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(title.strip() + "\n")


def is_duplicate_headline(title: str, processed_normalized: List[str]) -> bool:
    norm = _normalize_headline(title)
    if norm in processed_normalized:
        return True
    window = processed_normalized[-FUZZY_DEDUP_MAX_COMPARE:]
    for prev in window:
        if difflib.SequenceMatcher(None, norm, prev).ratio() >= FUZZY_DEDUP_THRESHOLD:
            return True
    return False


def load_queue(path: str = QUEUED_NEWS_FILE) -> List[NewsItem]:
    if not os.path.exists(path):
        return []
    items: List[NewsItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                items.append(NewsItem(
                    title=d["title"], summary=d.get("summary", ""),
                    link=d.get("link", ""), source_name=d.get("source_name", ""),
                ))
            except Exception as e:
                log.warning(f"  Skipping malformed queue line: {e}")
    return items


def save_queue(items: List[NewsItem], path: str = QUEUED_NEWS_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps({
                "title": it.title, "summary": it.summary,
                "link": it.link, "source_name": it.source_name,
            }, ensure_ascii=False) + "\n")


def _debug_dump_entry(entry) -> None:
    global _rss_debug_dumped
    if _rss_debug_dumped:
        return
    _rss_debug_dumped = True
    try:
        src = getattr(entry, "source", None)
        log.info("  [DEBUG] Sample RSS entry structure (first scanned item this process):")
        log.info(f"    title        = {getattr(entry, 'title', None)!r}")
        log.info(f"    link         = {str(getattr(entry, 'link', None))[:90]!r}")
        if src is not None:
            log.info(f"    source.title = {getattr(src, 'title', None)!r}")
            log.info(f"    source.href  = {getattr(src, 'href', None)!r}")
        else:
            log.info("    source       = <not present on this entry - title-suffix fallback will be used>")
        log.info("    (Set DEBUG_RSS_STRUCTURE = False once verified, to reduce log noise.)")
    except Exception as e:
        log.warning(f"  [DEBUG] entry dump failed (non-fatal): {e}")


def _extract_source_name(entry) -> str:
    src = getattr(entry, "source", None)
    if src is not None:
        title = getattr(src, "title", None)
        if not title and isinstance(src, dict):
            title = src.get("title")
        if title:
            return str(title).strip()
    raw_title = getattr(entry, "title", "") or ""
    if " - " in raw_title:
        return raw_title.rsplit(" - ", 1)[-1].strip()
    return ""


def _is_whitelisted_source(source_name: str, link: str) -> bool:
    """Primary signal is source_name; the link-domain check is a harmless
    secondary signal only (Google News RSS `link` values are normally
    redirect blobs, not the publisher's real domain). Verify with
    DEBUG_RSS_STRUCTURE=True on first run."""
    name_lower = (source_name or "").lower()
    link_lower = (link or "").lower()
    for domain, fragments in SOURCE_WHITELIST.items():
        if domain in link_lower:
            return True
        for frag in fragments:
            if frag in name_lower:
                return True
    return False


def scan_fresh_candidates(
    processed_normalized: List[str],
    exclude_items: List[NewsItem],
    max_count: int = MAX_CANDIDATES_TO_SCORE,
) -> List[NewsItem]:
    """Scans the RSS feed for fresh, whitelisted headlines not already in the
    processed log or in the current queue/exclude list."""
    log.info("Scanning Google News RSS for fresh, whitelisted candidate headlines ...")
    exclude_norm = [_normalize_headline(it.title) for it in exclude_items]

    feed = feedparser.parse(NEWS_RSS_URL)
    if not feed.entries:
        log.warning("  RSS feed returned no entries.")
        return []

    results: List[NewsItem] = []
    for entry in feed.entries[:MAX_FEED_ENTRIES_SCANNED]:
        if DEBUG_RSS_STRUCTURE:
            _debug_dump_entry(entry)

        title = _strip_html(getattr(entry, "title", "")).strip()
        if not title or len(title) < 5:
            continue
        if is_duplicate_headline(title, processed_normalized) or is_duplicate_headline(title, exclude_norm):
            continue

        source_name = _extract_source_name(entry)
        link = getattr(entry, "link", "")
        if not _is_whitelisted_source(source_name, link):
            continue

        summary = _strip_html(getattr(entry, "summary", "")).strip()
        if not summary or summary == title:
            summary = title

        results.append(NewsItem(title=title, summary=summary, link=link, source_name=source_name))
        if len(results) >= max_count:
            break

    log.info(f"  Found {len(results)} fresh whitelisted candidate(s).")
    return results


# ==============================================================================
# 5. GEMINI: CANDIDATE RANKING (1 call) + FACT-CHECK/SCRIPT (per story)
# ==============================================================================
SCORING_PROMPT_TEMPLATE = """You are a viral tech content strategist for a short-form
AI & Technology video channel (AI tools, productivity, gadgets, big tech news).

Rank the following candidate news stories by viral potential for a short-form video
(curiosity, surprise, relevance to an AI/tech-interested audience, shareability).

STORIES:
{numbered_list}

Respond ONLY with a valid JSON array, one object per story, in this exact form:
[{{"index": 0, "virality_score": 7, "reason": "one short phrase"}}, ...]
Include every index from 0 to {max_index} exactly once. virality_score is an integer 1-10.
"""

GEMINI_PROMPT_TEMPLATE = """You are an expert viral tech scriptwriter for a global/US English
YouTube Shorts / Instagram Reels / TikTok channel covering AI & Technology (AI tools,
productivity, gadgets, big tech news).

ROLE 1 - FACT-CHECKER: Judge whether this headline/context is a credible, verifiable tech
news story (as opposed to an unverified rumor, marketing hype dressed as news, or baseless
clickbait).

ROLE 2 - VIRAL SCRIPTWRITER: If credible, write a 30-40 second vertical video script package.

NEWS HEADLINE: {title}
NEWS SOURCE: {source_name}
NEWS CONTEXT: {summary}
{feedback_block}
Respond ONLY with a single valid JSON object (no markdown fences, no commentary) with
EXACTLY these keys:

{{
  "is_credible": true or false,
  "credibility_note": "One short sentence explaining the credibility judgement.",
  "script_en": "The full spoken voiceover script, 100% natural, simple, conversational
      English. STRICTLY 80 to 100 words so spoken audio lands between 30 and 40 seconds.
      Structure internally as: (0-3s) a shocking, curiosity-driven HOOK that stops the
      scroll and creates immediate suspense; (3-25s) fast-paced storytelling that keeps
      tension and curiosity high, written so the visual background could reasonably cut
      to something new every 2-3 seconds; (25-35s) a distinct ANALYSIS beat - your own
      one or two-sentence take on why this actually matters, who it affects, or what
      happens next. This must read as genuine commentary, not a repeated fact from the
      hook - this beat is what separates original commentary from a plain news reading;
      (last 3-5s) a punchy call-to-action telling viewers to like the video and subscribe
      so they never miss a tech update. Do NOT include stage directions, timestamps, or
      brackets - only words to be spoken aloud. If is_credible is false, still fill this
      with an empty string.",
  "title": "A highly clickbait, curiosity-driven video title in English with 1-3 emojis.",
  "description": "An SEO-optimized description (2-4 sentences) for YT Shorts/IG/TikTok, in English.",
  "hashtags": ["10 to 15 high-ranking hashtags as strings, each starting with #, e.g. #Shorts, #AI, #TechNews"],
  "tags": ["10 to 15 plain SEO keyword phrases (no #) for the YouTube tags box"],
  "thumbnail_idea": "One concise sentence describing a scroll-stopping thumbnail visual concept.",
  "visual_keywords": ["4 to 6 SIMPLE ENGLISH keywords/phrases describing generic stock
      footage that would visually match this story (e.g. 'server room', 'coding on
      laptop', 'robot arm factory', 'smartphone close up'). Generic and visual, not
      proper nouns."]
}}

Rules:
- script_en must be natural spoken English, energetic tone, no unexplained jargon.
- Output must be valid JSON only.
"""


def _extract_json_block(raw_text: str) -> str:
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()
    match = re.search(r"[\[{][\s\S]*[\]}]", text)
    if not match:
        raise ValueError("No JSON object/array found in Gemini response.")
    return match.group(0)


def _gemini_pace() -> None:
    global _last_gemini_call_ts
    elapsed = time.time() - _last_gemini_call_ts
    if elapsed < GEMINI_MIN_INTERVAL_SEC:
        time.sleep(GEMINI_MIN_INTERVAL_SEC - elapsed)
    _last_gemini_call_ts = time.time()


def score_candidate_stories(candidates: List[NewsItem]) -> List[Tuple[NewsItem, int]]:
    """Single Gemini call that ranks all candidates by viral potential.
    Falls back to feed order (earlier = higher) if scoring fails for any
    reason, so a Gemini hiccup never blocks the whole daily batch."""
    if not candidates:
        return []

    numbered = "\n".join(f"{i}. {c.title} — {c.summary[:160]}" for i, c in enumerate(candidates))
    prompt = SCORING_PROMPT_TEMPLATE.format(numbered_list=numbered, max_index=len(candidates) - 1)

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"temperature": 0.4, "response_mime_type": "application/json"},
    )

    scores: Dict[int, int] = {}
    try:
        _gemini_pace()
        resp = model.generate_content(prompt)
        data = json.loads(_extract_json_block(resp.text))
        for item in data:
            idx = int(item["index"])
            scores[idx] = int(item.get("virality_score", 5))
    except Exception as e:
        log.warning(f"  Story scoring failed ({e}); falling back to feed order.")
        scores = {i: (len(candidates) - i) for i in range(len(candidates))}

    scored = [(c, scores.get(i, 0)) for i, c in enumerate(candidates)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    log.info("  Ranked candidates: " + ", ".join(f"[{s}] {c.title[:40]}" for c, s in scored[:5]) + " ...")
    return scored


def generate_script_package(
    news: NewsItem,
    feedback: Optional[str] = None,
    budget: Optional[GeminiCallBudget] = None,
) -> ScriptPackage:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(
        GEMINI_MODEL,
        generation_config={"temperature": 0.9, "top_p": 0.95, "response_mime_type": "application/json"},
    )

    feedback_block = f"\nDURATION FEEDBACK FROM PREVIOUS ATTEMPT: {feedback}\n" if feedback else ""
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        title=news.title, source_name=news.source_name or "unknown",
        summary=news.summary, feedback_block=feedback_block,
    )

    required_keys = {
        "is_credible", "credibility_note", "script_en", "title", "description",
        "hashtags", "tags", "thumbnail_idea", "visual_keywords",
    }

    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_GEMINI_RETRIES + 1):
        if budget is not None and not budget.consume():
            raise RuntimeError(f"Gemini call budget exhausted ({budget.max_calls} calls) for this story.")
        try:
            _gemini_pace()
            response = model.generate_content(prompt)
            json_str = _extract_json_block(response.text)
            data = json.loads(json_str)

            missing = required_keys - set(data.keys())
            if missing:
                raise ValueError(f"Gemini JSON missing keys: {missing}")
            if not isinstance(data["hashtags"], list) or not isinstance(data["tags"], list):
                raise ValueError("hashtags/tags must be arrays.")

            is_credible = bool(data["is_credible"])
            script_en = str(data["script_en"]).strip()
            if is_credible and len(script_en) < 20:
                raise ValueError("script_en too short for a credible story - likely a bad generation.")

            return ScriptPackage(
                script_en=script_en,
                title=str(data["title"]).strip(),
                description=str(data["description"]).strip(),
                hashtags=[str(h).strip() for h in data["hashtags"] if str(h).strip()],
                tags=[str(t).strip() for t in data["tags"] if str(t).strip()],
                thumbnail_idea=str(data["thumbnail_idea"]).strip(),
                visual_keywords=[str(v).strip() for v in data.get("visual_keywords", []) if str(v).strip()],
                is_credible=is_credible,
                credibility_note=str(data.get("credibility_note", "")).strip(),
            )
        except Exception as e:  # noqa: BLE001
            last_error = e
            log.warning(f"  Gemini attempt {attempt}/{MAX_GEMINI_RETRIES} failed: {e}")

    raise RuntimeError(f"Gemini script generation failed after retries: {last_error}")


def generate_with_duration_enforcement(
    news: NewsItem,
    budget: Optional[GeminiCallBudget] = None,
) -> Tuple[ScriptPackage, str, List[WordTiming], float]:
    log.info(f"  Fact-check + script generation for: {news.title[:70]}")
    feedback: Optional[str] = None
    best: Optional[Tuple[float, ScriptPackage, str, List[WordTiming], float]] = None

    for attempt in range(1, MAX_DURATION_RETRIES + 1):
        if budget is not None and budget.remaining() <= 0:
            log.warning("  Gemini call budget exhausted before duration target reached; stopping retries.")
            break
        try:
            pkg = generate_script_package(news, feedback=feedback, budget=budget)
        except RuntimeError as e:
            log.warning(f"  {e}")
            break

        if not pkg.is_credible:
            raise CredibilitySkipError(pkg.credibility_note or "Flagged as unverified by fact-check step.")

        tmp_voice_path = os.path.join(TEMP_DIR, f"voice_attempt_{attempt}_{random.randint(1000,9999)}.mp3")
        words = synthesize_voice_with_timing(pkg.script_en, TTS_VOICE, tmp_voice_path)

        probe = AudioFileClip(tmp_voice_path)
        duration = probe.duration
        probe.close()

        diff = abs(duration - (TARGET_MIN_SEC + TARGET_MAX_SEC) / 2)
        if best is None or diff < best[0]:
            best = (diff, pkg, tmp_voice_path, words, duration)

        if TARGET_MIN_SEC <= duration <= TARGET_MAX_SEC:
            log.info(f"  -> Duration OK on attempt {attempt}: {duration:.1f}s")
            return pkg, tmp_voice_path, words, duration

        log.warning(f"  Attempt {attempt}/{MAX_DURATION_RETRIES}: duration {duration:.1f}s out of [30,40]s range, retrying ...")
        feedback = (
            f"The previous script produced {duration:.1f} seconds of speech, which is "
            f"{'too short' if duration < TARGET_MIN_SEC else 'too long'}. Rewrite the script to "
            f"strictly use 80 to 100 words so spoken duration lands between 30 and 40 seconds."
        )

    if best is None:
        raise RuntimeError("Could not generate a usable script within the Gemini call budget.")

    _, pkg, voice_path, words, duration = best
    log.warning(f"  Duration enforcement did not converge; proceeding with closest attempt ({duration:.1f}s).")
    return pkg, voice_path, words, duration


# ==============================================================================
# 6. VOICEOVER SYNTHESIS + CAPTION CHUNKING
# ==============================================================================
def synthesize_voice_with_timing(text: str, voice: str, out_mp3_path: str) -> List[WordTiming]:
    async def _run() -> List[WordTiming]:
        communicate = edge_tts.Communicate(text, voice, rate=TTS_RATE)
        words: List[WordTiming] = []
        with open(out_mp3_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 1e7
                    dur = chunk["duration"] / 1e7
                    words.append(WordTiming(text=chunk["text"], start=start, end=start + dur))
        return words

    words = asyncio.run(_run())
    if not os.path.exists(out_mp3_path) or os.path.getsize(out_mp3_path) == 0:
        raise RuntimeError("edge-tts produced no audio output. Check network/voice name.")
    if not words:
        log.warning("  No word-boundary timestamps returned; captions will be coarse.")
    return words


# ==============================================================================
# 7. PEXELS FOOTAGE SEARCH & DOWNLOAD
# ==============================================================================
def search_pexels_clips(query: str, per_page: int = PEXELS_RESULTS_PER_QUERY) -> List[str]:
    url = "https://api.pexels.com/videos/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "orientation": "portrait", "per_page": per_page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        videos = resp.json().get("videos", [])
    except Exception as e:
        log.warning(f"  Pexels search failed for '{query}': {e}")
        return []

    if not videos:
        params.pop("orientation", None)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            videos = resp.json().get("videos", [])
        except Exception as e:
            log.warning(f"  Pexels fallback search failed for '{query}': {e}")
            return []

    links = []
    for v in videos:
        files = [f for f in v.get("video_files", []) if f.get("width") and f.get("height")]
        if not files:
            continue
        files.sort(key=lambda f: abs((f["width"] or 0) - VIDEO_WIDTH))
        best = files[0]
        if best.get("link"):
            links.append(best["link"])
    return links


def download_file(url: str, dest_path: str) -> Optional[str]:
    try:
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        return dest_path
    except Exception as e:
        log.warning(f"  Download failed for {url}: {e}")
        return None


def fetch_background_clips(visual_keywords: List[str], fallback_query: str) -> List[str]:
    log.info("  Searching & downloading Pexels stock footage ...")
    queries = list(visual_keywords) if visual_keywords else []
    if fallback_query:
        queries.append(fallback_query)
    if not queries:
        queries = ["technology abstract", "data center servers"]

    seen_urls = set()
    local_paths: List[str] = []
    for q in queries:
        if len(local_paths) >= MAX_BACKGROUND_CLIPS:
            break
        for link in search_pexels_clips(q):
            if link in seen_urls:
                continue
            seen_urls.add(link)
            dest = os.path.join(TEMP_DIR, f"clip_{len(local_paths)}_{random.randint(1000,9999)}.mp4")
            path = download_file(link, dest)
            if path:
                local_paths.append(path)
            if len(local_paths) >= MAX_BACKGROUND_CLIPS:
                break

    if not local_paths:
        raise RuntimeError(
            "No background clips could be downloaded from Pexels. "
            "Check PEXELS_API_KEY and network connectivity."
        )
    return local_paths


# ==============================================================================
# 8. VIDEO ASSEMBLY (crop/resize/concat, center-weighted bounding)
# ==============================================================================
def crop_and_resize_9x16(clip: VideoFileClip) -> VideoFileClip:
    target_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    w, h = clip.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid clip dimensions.")
    current_ratio = w / h

    if current_ratio > target_ratio:
        new_w = int(h * target_ratio)
        x1 = max(0, (w - new_w) // 2)
        clip = clip.crop(x1=x1, y1=0, x2=x1 + new_w, y2=h)
    else:
        new_h = int(w / target_ratio)
        trimmed = h - new_h
        upward_bias = int(trimmed * 0.15)
        y1 = max(0, (trimmed // 2) - upward_bias)
        y1 = min(y1, h - new_h)
        clip = clip.crop(x1=0, y1=y1, x2=w, y2=y1 + new_h)

    return clip.resize((VIDEO_WIDTH, VIDEO_HEIGHT))


def _apply_ken_burns(clip: VideoFileClip, zoom_amount: float = KEN_BURNS_ZOOM_AMOUNT) -> VideoFileClip:
    """Slow continuous zoom-in over the clip's duration, output size fixed at
    the clip's own W x H (crops the zoomed frame back each frame) - a cheap
    way to avoid dead-static stock-footage frames without real motion
    tracking. Implemented via .fl() (per-frame PIL resize+crop) rather than
    moviepy's vfx.resize, because vfx.resize with a time-varying factor
    changes frame size over time, which CompositeVideoClip can't composite
    against a fixed canvas."""
    w, h = clip.size
    duration = max(clip.duration, 0.01)

    def make_frame(get_frame, t):
        frame = get_frame(t)
        progress = min(1.0, t / duration)
        scale = 1.0 + zoom_amount * progress
        new_w, new_h = max(w, int(w * scale)), max(h, int(h * scale))
        img = Image.fromarray(frame).resize((new_w, new_h), Image.LANCZOS)
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        img = img.crop((x1, y1, x1 + w, y1 + h))
        return np.array(img)

    return clip.fl(make_frame)


def apply_color_grade(clip: VideoFileClip) -> VideoFileClip:
    """Consistent, subtle color treatment applied to every video for a
    recognizable channel 'look' - a lightweight, zero-cost stand-in for a
    proper color-graded LUT. Never fatal: falls back to the ungraded clip on
    any error, since a missing 'look' is cosmetic, not a broken video."""
    try:
        graded = clip.fx(vfx.colorx, COLOR_GRADE_SATURATION)
        graded = graded.fx(vfx.lum_contrast, contrast=COLOR_GRADE_CONTRAST)
        return graded
    except Exception as e:
        log.warning(f"  Color grading skipped: {e}")
        return clip


def build_background_video(clip_paths: List[str], target_duration: float) -> VideoFileClip:
    log.info("  Assembling 1080x1920 background video ...")
    if target_duration <= 0:
        raise ValueError("target_duration must be positive.")

    segments = []
    total = 0.0
    idx = 0
    safety_counter = 0
    max_iterations = 500

    while total < target_duration and safety_counter < max_iterations:
        safety_counter += 1
        path = clip_paths[idx % len(clip_paths)]
        idx += 1
        try:
            raw_clip = VideoFileClip(path)
        except Exception as e:
            log.warning(f"  Skipping unreadable clip {path}: {e}")
            continue

        if raw_clip.duration is None or raw_clip.duration < 0.5:
            raw_clip.close()
            continue

        seg_len = random.uniform(CLIP_SEGMENT_MIN_SEC, CLIP_SEGMENT_MAX_SEC)
        seg_len = min(seg_len, target_duration - total, raw_clip.duration)
        if seg_len <= 0.15:
            raw_clip.close()
            continue

        max_start = max(0.0, raw_clip.duration - seg_len)
        start = random.uniform(0, max_start) if max_start > 0 else 0.0

        try:
            sub = raw_clip.subclip(start, start + seg_len)
            sub = crop_and_resize_9x16(sub)
            sub = sub.without_audio()
            if ENABLE_KEN_BURNS:
                sub = _apply_ken_burns(sub)
        except Exception as e:
            log.warning(f"  Failed to process segment from {path}: {e}")
            raw_clip.close()
            continue

        segments.append(sub)
        total += sub.duration

    if not segments:
        raise RuntimeError("Failed to build any usable video segments from downloaded clips.")

    if len(segments) > 1 and CROSSFADE_DURATION > 0:
        faded = [segments[0]] + [c.crossfadein(CROSSFADE_DURATION) for c in segments[1:]]
        bg = concatenate_videoclips(faded, padding=-CROSSFADE_DURATION, method="compose")
    else:
        bg = concatenate_videoclips(segments, method="compose")

    if bg.duration < target_duration:
        loops_needed = int(target_duration // bg.duration) + 1
        bg = concatenate_videoclips([bg] * loops_needed, method="compose")
    bg = bg.subclip(0, target_duration)
    bg = apply_color_grade(bg)

    log.info(f"  -> Background video assembled: {bg.duration:.1f}s across {len(segments)} segments")
    return bg


# ==============================================================================
# 9. DYNAMIC PIL CAPTIONS (English - no Devanagari/ImageMagick dependency)
# ==============================================================================
def find_caption_font() -> Optional[str]:
    """English captions can use any standard sans-serif font. DejaVu Sans Bold
    ships with most Linux distros/Colab images by default, so (unlike the old
    Devanagari path) no extra apt-get font package is required."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
        os.path.join(TEMP_DIR, "Roboto-Bold.ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            log.info(f"  Using caption font: {path}")
            return path

    dest = os.path.join(TEMP_DIR, "Roboto-Bold.ttf")
    url = "https://github.com/googlefonts/roboto/raw/main/src/hinted/Roboto-Bold.ttf"
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        log.info("  -> Downloaded Roboto-Bold font for captions.")
        return dest
    except Exception as e:
        log.warning(f"  Could not fetch a fallback font ({e}); captions will use PIL's default bitmap font.")
        return None


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, stroke_width: int = 4) -> Tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _layout_caption_words(
    words: List[str],
    font: ImageFont.FreeTypeFont,
    content_width: int,
    draw: ImageDraw.ImageDraw,
    stroke_width: int = 4,
) -> Tuple[List[List[dict]], int]:
    """Word-wraps a line of words to content_width using actual pixel widths
    (not character counts, which are unreliable for variable-width fonts).
    Returns (lines, space_width) where each line is a list of
    {'word', 'w', 'h'} dicts in reading order."""
    space_w, _ = _measure_text(draw, " ", font, stroke_width)
    lines: List[List[dict]] = []
    current: List[dict] = []
    current_w = 0
    for word in words:
        ww, wh = _measure_text(draw, word, font, stroke_width)
        extra = space_w if current else 0
        if current and current_w + extra + ww > content_width:
            lines.append(current)
            current, current_w, extra = [], 0, 0
        current.append({"word": word, "w": ww, "h": wh})
        current_w += extra + ww
    if current:
        lines.append(current)
    return lines, space_w


def render_caption_line_highlighted(
    line_words: List[str],
    highlight_idx: Optional[int],
    font_path: Optional[str],
    width: int = VIDEO_WIDTH,
    font_size: int = 66,
) -> np.ndarray:
    """Renders one caption line with every word in white except
    highlight_idx (drawn in accent yellow) - the 'karaoke' look. Called once
    per WORD with the same line_words/layout, only highlight_idx changes, so
    the caption bar never jitters between words - only the color does."""
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    probe = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    d = ImageDraw.Draw(probe)
    bar_margin = 24
    content_width = width - 2 * bar_margin - 60

    lines, space_w = _layout_caption_words(line_words, font, content_width, d)
    line_heights = [max((wd["h"] for wd in line), default=font_size) for line in lines]
    padding_y, line_spacing = 24, 14
    block_height = sum(line_heights) + line_spacing * (len(lines) - 1) + padding_y * 2

    img = Image.new("RGBA", (width, block_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(
        [bar_margin, 0, width - bar_margin, block_height], radius=20, fill=(0, 0, 0, 140),
    )

    global_idx = 0
    y = padding_y
    for line, lh in zip(lines, line_heights):
        line_w = sum(wd["w"] for wd in line) + space_w * (len(line) - 1)
        x = (width - line_w) // 2
        for wd in line:
            is_current = (highlight_idx is not None and global_idx == highlight_idx)
            color = (255, 220, 0, 255) if is_current else (255, 255, 255, 255)
            draw.text(
                (x, y), wd["word"], font=font,
                fill=color, stroke_width=4, stroke_fill=(0, 0, 0, 255),
            )
            x += wd["w"] + space_w
            global_idx += 1
        y += lh + line_spacing

    return np.array(img)


def _build_caption_chunks_fallback(audio_duration: float, full_script: str) -> List[CaptionChunk]:
    """Used only when edge-tts returned no word-boundary timestamps (rare) -
    evenly spaces static (non-karaoke) chunks across the audio so captions
    still exist, just without word-level highlighting."""
    pieces = full_script.split()
    chunk_size = max(1, WORDS_PER_CAPTION_CHUNK)
    n_chunks = max(1, len(pieces) // chunk_size + (1 if len(pieces) % chunk_size else 0))
    seg_len = audio_duration / n_chunks
    chunks = []
    for i in range(n_chunks):
        text = " ".join(pieces[i * chunk_size:(i + 1) * chunk_size])
        if text:
            chunks.append(CaptionChunk(text=text, start=i * seg_len, end=(i + 1) * seg_len))
    return chunks


def _compute_word_display_ends(words: List[WordTiming], audio_duration: float) -> List[float]:
    """Extends each word's display end time to the next word's start (so a
    caption never blanks out during a natural TTS pause), and the final
    word's end to the full audio duration. Computed across the WHOLE word
    sequence, independent of how words are later grouped into lines - a
    pause that falls on a line boundary still gets bridged correctly."""
    ends: List[float] = []
    for i, w in enumerate(words):
        if i + 1 < len(words):
            ends.append(max(w.end, words[i + 1].start))
        else:
            ends.append(max(w.end, audio_duration))
    return ends


def build_caption_clips(
    words: List[WordTiming],
    audio_duration: float,
    full_script: str,
    font_path: Optional[str],
) -> List[ImageClip]:
    """Karaoke-style captions: groups words into short on-screen lines
    (WORDS_PER_CAPTION_CHUNK words each) and renders ONE ImageClip per WORD
    within that line - same line text/layout throughout the line's time
    span, only the currently-spoken word recolored - instead of one static
    block of text per line."""
    log.info("  Rendering karaoke-style word-highlighted captions ...")
    clips: List[ImageClip] = []

    if not words:
        for chunk in _build_caption_chunks_fallback(audio_duration, full_script):
            try:
                arr = render_caption_line_highlighted(chunk.text.split(), None, font_path)
            except Exception as e:
                log.warning(f"  Failed to render fallback caption '{chunk.text[:20]}...': {e}")
                continue
            duration = max(0.15, chunk.end - chunk.start)
            clips.append(
                ImageClip(arr).set_start(chunk.start).set_duration(duration)
                .set_position(("center", int(VIDEO_HEIGHT * 0.72)))
            )
        return clips

    display_ends = _compute_word_display_ends(words, audio_duration)

    for i in range(0, len(words), WORDS_PER_CAPTION_CHUNK):
        group = words[i:i + WORDS_PER_CAPTION_CHUNK]
        group_ends = display_ends[i:i + WORDS_PER_CAPTION_CHUNK]
        line_words = [w.text for w in group]
        for local_idx, (w, w_end) in enumerate(zip(group, group_ends)):
            duration = max(0.08, w_end - w.start)
            try:
                arr = render_caption_line_highlighted(line_words, local_idx, font_path)
            except Exception as e:
                log.warning(f"  Failed to render caption word '{w.text}': {e}")
                continue
            clips.append(
                ImageClip(arr).set_start(w.start).set_duration(duration)
                .set_position(("center", int(VIDEO_HEIGHT * 0.72)))
            )
    return clips


def _ensure_shorts_hashtag(pkg: ScriptPackage) -> None:
    """YouTube routes a video into the Shorts shelf using duration (<=60s,
    already satisfied here) and vertical aspect ratio (already 1080x1920) as
    the main signals, but including '#Shorts' in the title/description is a
    well-documented extra nudge. Belt-and-suspenders in case Gemini forgot it."""
    if not any(h.strip().lower() == "#shorts" for h in pkg.hashtags):
        pkg.hashtags.insert(0, "#Shorts")


def build_watermark_clip(duration: float) -> Optional[ImageClip]:
    """Small semi-transparent logo in the top-right corner for the whole
    main-content duration. Skips gracefully (returns None) if LOGO_PATH
    doesn't exist - branding is a nice-to-have, never a hard requirement."""
    if not os.path.exists(LOGO_PATH):
        return None
    try:
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        target_w = int(VIDEO_WIDTH * WATERMARK_WIDTH_RATIO)
        ratio = target_w / logo_img.width
        target_h = max(1, int(logo_img.height * ratio))
        logo_img = logo_img.resize((target_w, target_h), Image.LANCZOS)

        alpha = logo_img.split()[3].point(lambda p: int(p * WATERMARK_OPACITY))
        logo_img.putalpha(alpha)

        margin = 32
        return (
            ImageClip(np.array(logo_img))
            .set_duration(duration)
            .set_position((VIDEO_WIDTH - target_w - margin, margin))
        )
    except Exception as e:
        log.warning(f"  Watermark generation skipped: {e}")
        return None


def build_intro_sting() -> Optional[ImageClip]:
    """A short branded opening: the logo fading in/out on a black background.
    Adds INTRO_STING_DURATION seconds ON TOP of the 30-40s main content -
    total stays comfortably under YouTube's Shorts duration limit. Skips
    gracefully if LOGO_PATH doesn't exist."""
    if not os.path.exists(LOGO_PATH):
        return None
    try:
        logo_img = Image.open(LOGO_PATH).convert("RGBA")
        target_w = int(VIDEO_WIDTH * 0.45)
        ratio = target_w / logo_img.width
        target_h = max(1, int(logo_img.height * ratio))
        logo_img = logo_img.resize((target_w, target_h), Image.LANCZOS)

        black_bg = Image.new("RGBA", (VIDEO_WIDTH, VIDEO_HEIGHT), (0, 0, 0, 255))
        x = (VIDEO_WIDTH - target_w) // 2
        y = (VIDEO_HEIGHT - target_h) // 2
        black_bg.paste(logo_img, (x, y), logo_img)

        return (
            ImageClip(np.array(black_bg))
            .set_duration(INTRO_STING_DURATION)
            .fadein(0.25)
            .fadeout(0.2)
        )
    except Exception as e:
        log.warning(f"  Intro sting generation skipped: {e}")
        return None


def pick_background_music(duration: float) -> Optional[AudioFileClip]:
    """Picks a random track from MUSIC_DIR, loops it to cover `duration`, and
    lowers its volume to MUSIC_VOLUME so it sits under the voice, not over
    it. Returns None (no music) if MUSIC_DIR doesn't exist or is empty -
    this is expected until you drop your own royalty-free tracks in there."""
    if not os.path.isdir(MUSIC_DIR):
        return None
    candidates = [f for f in os.listdir(MUSIC_DIR) if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg"))]
    if not candidates:
        return None

    chosen = os.path.join(MUSIC_DIR, random.choice(candidates))
    try:
        music = AudioFileClip(chosen)
    except Exception as e:
        log.warning(f"  Could not load background music '{chosen}': {e}")
        return None

    try:
        if music.duration < duration:
            loops_needed = int(duration // music.duration) + 1
            music = concatenate_audioclips([music] * loops_needed)
        music = music.subclip(0, duration).volumex(MUSIC_VOLUME)
        return music
    except Exception as e:
        log.warning(f"  Background music processing failed, continuing without music: {e}")
        return None


def generate_thumbnail(video_path: str, pkg: ScriptPackage, font_path: Optional[str]) -> Optional[str]:
    """Zero-cost thumbnail: grabs a frame a little into the video (skipping
    the very first frame, which is often mid-transition) and overlays a
    short, bold headline using the same PIL text pipeline as the captions -
    no paid image-generation API needed. Output is the native 1080x1920
    vertical frame, since YouTube's own Shorts thumbnail guidance recommends
    9:16 (not the classic 16:9 long-form thumbnail size).

    HONEST CAVEAT: YouTube's Shorts swipe feed usually IGNORES a custom
    thumbnail and auto-picks its own frame regardless of what's uploaded here
    - this is a platform limitation, not a bug in this script. The custom
    thumbnail still shows in search results, the channel's video grid, and
    "Watch Later"/playlists, so it's not wasted, just not guaranteed to
    appear in the Shorts feed itself."""
    try:
        clip = VideoFileClip(video_path)
        grab_time = min(1.2, max(0.1, clip.duration * 0.06))
        frame = clip.get_frame(grab_time)
        clip.close()
    except Exception as e:
        log.warning(f"  Thumbnail frame extraction failed: {e}")
        return None

    try:
        img = Image.fromarray(frame).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")
        w, h = img.size

        headline_plain = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF]", "", pkg.title).strip()
        wrapped = textwrap.fill(headline_plain[:70] or "Breaking Tech News", width=16)
        lines = wrapped.split("\n")

        try:
            font = ImageFont.truetype(font_path, 84) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=6)
            line_heights.append(bbox[3] - bbox[1])
        line_spacing = 16
        block_height = sum(line_heights) + line_spacing * (len(lines) - 1) + 80

        draw.rectangle([0, h - block_height, w, h], fill=(0, 0, 0, 165))

        y = h - block_height + 40
        for line, lh in zip(lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=6)
            lw = bbox[2] - bbox[0]
            x = (w - lw) // 2
            draw.text(
                (x, y), line, font=font,
                fill=(255, 220, 0, 255), stroke_width=6, stroke_fill=(0, 0, 0, 255),
            )
            y += lh + line_spacing

        thumb_path = os.path.join(TEMP_DIR, f"thumb_{random.randint(1000,9999)}.jpg")
        img.save(thumb_path, "JPEG", quality=90)
        return thumb_path
    except Exception as e:
        log.warning(f"  Thumbnail rendering failed: {e}")
        return None


def set_youtube_thumbnail(video_id: str, thumbnail_path: str) -> bool:
    try:
        youtube = get_youtube_service()
        youtube.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(thumbnail_path)).execute()
        log.info(f"  -> Custom thumbnail set for {video_id}")
        return True
    except Exception as e:
        log.warning(f"  Setting custom thumbnail failed (video still uploaded fine): {e}")
        return False


# ==============================================================================
# 10. FINAL VIDEO ASSEMBLY
# ==============================================================================
def assemble_final_video(
    bg_clip: VideoFileClip,
    caption_clips: List[ImageClip],
    audio_path: str,
    out_path: str,
) -> str:
    log.info("  Rendering final 1080p vertical video (this can take a while) ...")
    audio = AudioFileClip(audio_path)
    duration = audio.duration

    music = pick_background_music(duration)
    final_audio = CompositeAudioClip([audio, music]) if music is not None else audio

    bg_clip = bg_clip.set_duration(duration)
    watermark = build_watermark_clip(duration)
    layers = [bg_clip, *caption_clips] + ([watermark] if watermark is not None else [])

    main_content = CompositeVideoClip(layers, size=(VIDEO_WIDTH, VIDEO_HEIGHT)).set_duration(duration)
    main_content = main_content.set_audio(final_audio)

    intro = build_intro_sting()
    final = concatenate_videoclips([intro, main_content], method="compose") if intro is not None else main_content

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    final.write_videofile(
        out_path, fps=30, codec="libx264", audio_codec="aac",
        preset="medium", bitrate="6000k", threads=4, logger=None,
    )

    audio.close()
    if music is not None:
        music.close()
    final.close()
    log.info(f"  -> Final video saved: {out_path} ({final.duration:.1f}s total)")
    return out_path


# ==============================================================================
# 11. YOUTUBE: OAUTH, SCHEDULED UPLOAD, TIME-SLOT COMPUTATION
# ==============================================================================
YOUTUBE_UPLOAD_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]


def get_youtube_service():
    # Preferred path for CI: build credentials directly from a refresh token
    # obtained via OAuth Playground - no local browser flow, no token file.
    if YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET and YOUTUBE_REFRESH_TOKEN:
        creds = Credentials(
            token=None,
            refresh_token=YOUTUBE_REFRESH_TOKEN,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=YOUTUBE_CLIENT_ID,
            client_secret=YOUTUBE_CLIENT_SECRET,
            scopes=YOUTUBE_UPLOAD_SCOPES,
        )
        creds.refresh(GoogleAuthRequest())  # exchanges the refresh token for a fresh access token
        return build_google_service("youtube", "v3", credentials=creds)

    # Fallback path for local/Colab use: file-based token, refreshed or
    # created via a one-time browser consent flow.
    creds = None
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(YOUTUBE_TOKEN_FILE, YOUTUBE_UPLOAD_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(GoogleAuthRequest())
        else:
            if not os.path.exists(YOUTUBE_CLIENT_SECRETS_FILE):
                raise RuntimeError(
                    f"No YouTube credentials available: set YOUTUBE_CLIENT_ID + "
                    f"YOUTUBE_CLIENT_SECRET + YOUTUBE_REFRESH_TOKEN (CI), or provide "
                    f"'{YOUTUBE_CLIENT_SECRETS_FILE}' for a local browser-based flow. "
                    f"See the OAuth setup notes at the top of this file."
                )
            flow = InstalledAppFlow.from_client_secrets_file(YOUTUBE_CLIENT_SECRETS_FILE, YOUTUBE_UPLOAD_SCOPES)
            creds = flow.run_local_server(port=0)
        with open(YOUTUBE_TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return build_google_service("youtube", "v3", credentials=creds)


def _random_minute_in_window(start_h: int, start_m: int, end_h: int, end_m: int) -> Tuple[int, int]:
    start_total = start_h * 60 + start_m
    end_total = end_h * 60 + end_m
    span = max(1, end_total - start_total)
    target_total = start_total + random.randint(0, span - 1)
    return divmod(target_total, 60)


def compute_slot_datetime(slot_index: int) -> str:
    """Returns an RFC3339 UTC timestamp ('...Z') for a random minute inside
    the next available occurrence of the given slot window (0, 1, or 2),
    rolling to tomorrow if today's window is already within
    SLOT_MIN_LEAD_MINUTES of ending or fully in the past. Randomizing within
    the window (rather than a fixed HH:MM) avoids every video publishing at
    an identical, obviously-automated timestamp day after day."""
    tz = ZoneInfo(YOUTUBE_TIME_ZONE)
    now_local = datetime.now(tz)
    start_h, start_m, end_h, end_m = SLOT_WINDOWS[slot_index % len(SLOT_WINDOWS)]

    hour, minute = _random_minute_in_window(start_h, start_m, end_h, end_m)
    candidate = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)

    window_end_today = now_local.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
    if candidate <= now_local + timedelta(minutes=SLOT_MIN_LEAD_MINUTES) or window_end_today <= now_local + timedelta(minutes=SLOT_MIN_LEAD_MINUTES):
        hour, minute = _random_minute_in_window(start_h, start_m, end_h, end_m)
        candidate = (now_local + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)

    candidate_utc = candidate.astimezone(timezone.utc)
    return candidate_utc.strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate_youtube_tags(tags: List[str], limit_chars: int = 460) -> List[str]:
    out, total = [], 0
    for t in tags:
        add_len = len(t) + 1
        if total + add_len > limit_chars:
            break
        out.append(t)
        total += add_len
    return out


def upload_video_to_youtube(video_path: str, pkg: ScriptPackage, publish_at_utc: str) -> Optional[str]:
    """Uploads as privacyStatus=private with publishAt set, so YouTube flips
    it to public automatically at that timestamp - no manual step needed.
    Returns the video ID, or None if the upload was skipped/failed (caller
    treats this as non-fatal; the video still exists locally + on Telegram)."""
    try:
        youtube = get_youtube_service()
    except Exception as e:
        log.warning(f"  YouTube auth unavailable, skipping upload: {e}")
        return None

    body = {
        "snippet": {
            "title": pkg.title[:100],
            "description": (pkg.description + "\n\n" + " ".join(pkg.hashtags))[:4900],
            "tags": _truncate_youtube_tags(pkg.tags),
            "categoryId": YOUTUBE_CATEGORY_ID,
        },
        "status": {
            "privacyStatus": "private",
            "publishAt": publish_at_utc,
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": True,  # honest disclosure: TTS voice + AI-written script
        },
    }
    media = MediaFileUpload(video_path, chunksize=-1, resumable=True, mimetype="video/mp4")

    try:
        request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
        response = None
        while response is None:
            status, response = request.next_chunk()
        video_id = response.get("id")
        log.info(f"  -> Uploaded to YouTube, scheduled {publish_at_utc}: https://studio.youtube.com/video/{video_id}/edit")
        return video_id
    except Exception as e:
        log.warning(f"  YouTube upload request failed: {e}")
        return None


# ------------------------------------------------------------------------------
# YouTube-only retry queue: a story whose VIDEO rendered fine but whose
# YouTube upload failed (bad/expired OAuth, quota, transient API error) is
# never re-rendered from scratch - only the upload itself is retried, using
# the locally-saved mp4 (OUTPUT_DIR is never cleaned up between cycles).
# ------------------------------------------------------------------------------
def load_pending_youtube_uploads(path: str = PENDING_YOUTUBE_UPLOADS_FILE) -> List[dict]:
    if not os.path.exists(path):
        return []
    items: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                log.warning(f"  Skipping malformed pending-upload line: {e}")
    return items


def save_pending_youtube_uploads(items: List[dict], path: str = PENDING_YOUTUBE_UPLOADS_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def queue_pending_youtube_upload(video_path: str, pkg: ScriptPackage, slot_index: int) -> None:
    pending = load_pending_youtube_uploads()
    pending.append({
        "video_path": video_path,
        "title": pkg.title,
        "description": pkg.description,
        "hashtags": pkg.hashtags,
        "tags": pkg.tags,
        "slot_index": slot_index,
        "retry_count": 0,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    })
    save_pending_youtube_uploads(pending)
    log.info(f"  Queued for YouTube-only retry later: {pkg.title[:50]}")


def retry_pending_youtube_uploads() -> None:
    """Called at the start of every daily batch, before touching new
    candidates, so a fixed OAuth token or transient API outage gets
    yesterday's failed uploads out the door before spending any Gemini/Pexels
    budget on fresh stories. Entries missing their local file, or that have
    exhausted MAX_YOUTUBE_UPLOAD_RETRIES, are dropped (logged, not silently)."""
    pending = load_pending_youtube_uploads()
    if not pending:
        return

    log.info(f"Retrying {len(pending)} pending YouTube upload(s) from previous cycles ...")
    still_pending: List[dict] = []
    for entry in pending:
        video_path = entry.get("video_path", "")
        title = entry.get("title", "")[:60]

        if not os.path.exists(video_path):
            log.warning(f"  Dropping pending upload - local file no longer exists: {title}")
            continue
        if entry.get("retry_count", 0) >= MAX_YOUTUBE_UPLOAD_RETRIES:
            log.warning(f"  Dropping pending upload after {entry.get('retry_count', 0)} failed retries: {title}")
            continue

        pkg_stub = ScriptPackage(
            script_en="", title=entry.get("title", ""), description=entry.get("description", ""),
            hashtags=entry.get("hashtags", []), tags=entry.get("tags", []),
            thumbnail_idea="", visual_keywords=[],
        )
        publish_at = compute_slot_datetime(entry.get("slot_index", 0))

        video_id = None
        try:
            video_id = upload_video_to_youtube(video_path, pkg_stub, publish_at)
        except Exception as e:
            log.warning(f"  Retry upload raised an unexpected error: {e}")

        if video_id:
            log.info(f"  -> Pending upload succeeded on retry: {title}")
            thumb_path = generate_thumbnail(video_path, pkg_stub, find_caption_font())
            if thumb_path:
                set_youtube_thumbnail(video_id, thumb_path)
        else:
            entry["retry_count"] = entry.get("retry_count", 0) + 1
            still_pending.append(entry)

    save_pending_youtube_uploads(still_pending)
    if still_pending:
        log.info(f"  {len(still_pending)} upload(s) still pending after this retry pass.")


# ==============================================================================
# 12. TELEGRAM DELIVERY
# ==============================================================================
def send_video_to_telegram(video_path: str, caption: Optional[str] = None) -> dict:
    log.info("  Uploading video to Telegram ...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"

    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > 49:
        log.warning(f"  Video is {file_size_mb:.1f}MB - close to/over Telegram's 50MB Bot API upload limit.")

    with open(video_path, "rb") as f:
        files = {"video": (os.path.basename(video_path), f, "video/mp4")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "supports_streaming": True}
        if caption:
            data["caption"] = caption[:1024]
        resp = requests.post(url, data=data, files=files, timeout=600)
    if not resp.ok:
        raise RuntimeError(f"Telegram sendVideo failed: {resp.status_code} {resp.text}")
    log.info("  -> Video delivered to Telegram.")
    return resp.json()


def send_message_to_telegram(text: str) -> None:
    log.info("  Sending copy-paste metadata to Telegram ...")
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    for piece in textwrap.wrap(text, 3800, replace_whitespace=False, break_long_words=False):
        resp = requests.post(
            url, data={"chat_id": TELEGRAM_CHAT_ID, "text": piece, "disable_web_page_preview": True},
            timeout=60,
        )
        if not resp.ok:
            raise RuntimeError(f"Telegram sendMessage failed: {resp.status_code} {resp.text}")
    log.info("  -> Metadata delivered to Telegram.")


def format_metadata_message(pkg: ScriptPackage, news: NewsItem, publish_at_utc: Optional[str] = None) -> str:
    hashtags_line = " ".join(pkg.hashtags)
    tags_line = ", ".join(pkg.tags)
    schedule_line = f"\n\n🗓️ SCHEDULED (YouTube, UTC): {publish_at_utc}" if publish_at_utc else "\n\n🗓️ YouTube: not scheduled (upload skipped)"
    return (
        f"📌 TITLE:\n{pkg.title}\n\n"
        f"📝 DESCRIPTION:\n{pkg.description}\n\n"
        f"🏷️ HASHTAGS:\n{hashtags_line}\n\n"
        f"🔑 SEO TAGS:\n{tags_line}\n\n"
        f"💡 THUMBNAIL IDEA:\n{pkg.thumbnail_idea}\n\n"
        f"📰 SOURCE HEADLINE ({news.source_name}):\n{news.title}"
        f"{schedule_line}"
    )


# ==============================================================================
# 13. CLEANUP
# ==============================================================================
def cleanup_temp_files() -> None:
    try:
        import shutil
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        os.makedirs(TEMP_DIR, exist_ok=True)
    except Exception as e:
        log.warning(f"Cleanup warning: {e}")


# ==============================================================================
# 14. SINGLE-STORY ORCHESTRATION (called up to DAILY_STORY_TARGET times/batch)
# ==============================================================================
def process_single_story(news: NewsItem, slot_index: int) -> bool:
    """Full pipeline for one story: script -> voice -> footage -> captions ->
    render -> YouTube (scheduled) -> Telegram. Returns True on a produced,
    delivered video; False if skipped (fact-check) or failed before a local
    video file existed (safe to retry another day). Once the local video file
    exists, the headline is marked processed immediately - YouTube/Telegram
    delivery failures after that point are logged but never trigger a
    re-render of the same story."""
    budget = GeminiCallBudget(MAX_GEMINI_CALLS_PER_CYCLE)
    try:
        pkg, voice_path, words, audio_duration = generate_with_duration_enforcement(news, budget)
    except CredibilitySkipError as e:
        log.warning(f"  Skipping headline (fact-check flagged it): {e.note}")
        mark_headline_processed(news.title)
        return False
    except Exception:
        log.error(f"  Script generation failed for '{news.title[:60]}':")
        log.error(traceback.format_exc())
        return False

    _ensure_shorts_hashtag(pkg)

    try:
        font_path = find_caption_font()
        caption_clips = build_caption_clips(words, audio_duration, pkg.script_en, font_path)
        fallback_query = news.title.split(" - ")[0][:60]
        clip_paths = fetch_background_clips(pkg.visual_keywords, fallback_query)
        bg_video = build_background_video(clip_paths, audio_duration)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        safe_slug = re.sub(r"[^a-zA-Z0-9]+", "_", news.title.lower())[:40].strip("_")
        final_path = os.path.join(OUTPUT_DIR, f"short_{safe_slug or 'story'}.mp4")
        assemble_final_video(bg_video, caption_clips, voice_path, final_path)
    except Exception:
        log.error(f"  Video rendering failed for '{news.title[:60]}':")
        log.error(traceback.format_exc())
        return False

    # Local video file exists now - mark processed so delivery failures below
    # can never cause this story to be re-generated on a future cycle.
    mark_headline_processed(news.title)

    publish_at = compute_slot_datetime(slot_index)
    video_id = None
    try:
        video_id = upload_video_to_youtube(final_path, pkg, publish_at)
    except Exception as e:
        log.warning(f"  Unhandled YouTube upload error (video saved locally): {e}")

    if not video_id:
        queue_pending_youtube_upload(final_path, pkg, slot_index)
    else:
        thumb_path = generate_thumbnail(final_path, pkg, font_path)
        if thumb_path:
            set_youtube_thumbnail(video_id, thumb_path)

    try:
        short_caption = f"{pkg.title}\n\n{' '.join(pkg.hashtags[:8])}\n\nScheduled: {publish_at}"
        send_video_to_telegram(final_path, caption=short_caption)
        send_message_to_telegram(format_metadata_message(pkg, news, publish_at if video_id else None))
    except Exception as e:
        log.warning(f"  Telegram delivery failed (story already fully processed otherwise): {e}")

    log.info(f"  Story processed successfully: {news.title[:60]}")
    return True


# ==============================================================================
# 15. DAILY BATCH ORCHESTRATION
# ==============================================================================
def run_daily_batch() -> int:
    """One full daily cycle: gather candidates (queue + fresh RSS), rank them
    with a single Gemini call, produce up to DAILY_STORY_TARGET videos from
    the best of them (with backfill from the ranked pool if a top pick is
    skipped/fails), schedule each to YouTube's next available slot, deliver
    to Telegram, and persist the rest to queued_news.txt for future days."""
    retry_pending_youtube_uploads()

    processed = load_processed_headlines()
    queue_items = load_queue()
    fresh_items = scan_fresh_candidates(processed, exclude_items=queue_items)

    candidates = (queue_items + fresh_items)[:MAX_CANDIDATES_TO_SCORE]
    if not candidates:
        log.info("No candidate stories available (queue empty, no fresh whitelisted headlines).")
        save_queue([])
        return 0

    scored = score_candidate_stories(candidates)
    pool = [item for item, _score in scored]

    produced = 0
    slot_index = 0
    attempts = 0

    while produced < DAILY_STORY_TARGET and pool and attempts < MAX_DAILY_ATTEMPTS:
        attempts += 1
        news = pool.pop(0)
        try:
            ok = process_single_story(news, slot_index)
        except Exception:
            log.error(f"Unhandled failure processing '{news.title[:60]}':")
            log.error(traceback.format_exc())
            ok = False
        if ok:
            produced += 1
            slot_index += 1

    save_queue(pool)  # whatever's left unprocessed/unpicked goes back to the queue
    log.info(f"Daily batch complete: {produced}/{DAILY_STORY_TARGET} videos produced, "
             f"{len(pool)} candidate(s) requeued for future days.")
    return produced


# ==============================================================================
# 16. MAIN (single batch or infinite scheduling loop)
# ==============================================================================
def main() -> int:
    try:
        validate_config()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    except Exception as e:
        log.error(f"Startup validation failed: {e}")
        return 1

    if AUTO_LOOP_INTERVAL_HOURS <= 0:
        try:
            run_daily_batch()
            return 0
        except Exception:
            log.error("Daily batch failed with an unhandled exception:")
            log.error(traceback.format_exc())
            return 1
        finally:
            cleanup_temp_files()

    log.info(f"Continuous mode enabled: one batch every {AUTO_LOOP_INTERVAL_HOURS}h. Ctrl+C to stop.")
    while True:
        try:
            run_daily_batch()
        except Exception:
            log.error("Batch failed with an unhandled exception (will retry next cycle):")
            log.error(traceback.format_exc())
        finally:
            cleanup_temp_files()

        try:
            log.info(f"Sleeping {AUTO_LOOP_INTERVAL_HOURS}h until next batch ...")
            time.sleep(AUTO_LOOP_INTERVAL_HOURS * 3600)
        except KeyboardInterrupt:
            log.info("Interrupted by user. Exiting cleanly.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
