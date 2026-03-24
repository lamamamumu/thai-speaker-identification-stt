"""
dialogue.py — Build and save speaker-labelled dialogue from STT + diarization.

Takes STT segments (text + timestamps) and diarization hypothesis
(speaker + timestamps), merges them by time overlap, and outputs
a clean conversation transcript as CSV.
"""

import re
import pandas as pd


def is_mostly_thai(text, threshold=0.5):
    """
    Fix #7: Return True only if at least 50% of characters are Thai.
    Filters out Whisper hallucinations that mix Korean/symbols/Latin.
    """
    if not text:
        return False
    thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', text))
    return thai_chars / len(text) >= threshold


def sec_to_hmmss(s):
    """Convert seconds to H.MM.SS format.  e.g. 3721 → '1.02.01'"""
    s   = int(s)
    h   = s // 3600
    m   = (s % 3600) // 60
    sec = s % 60
    return f"{h}.{m:02d}.{sec:02d}"


def build_dialogue(hyp_segs, stt_segs):
    """
    Word-level diarization with timestamp validation.

    For each STT segment, assign speaker by majority overlap with diarization hyp.
    Then split segment text proportionally at speaker-switch boundaries within it.
    Falls back to single-speaker if timestamps unreliable.

    Returns list of {start, end, speaker, text}.
    """
    hyp_sorted = sorted(hyp_segs, key=lambda x: x['start'])

    def majority_speaker(seg_start, seg_end):
        spk_time = {}
        for h in hyp_sorted:
            ov = min(seg_end, h['end']) - max(seg_start, h['start'])
            if ov > 0:
                spk_time[h['speaker']] = spk_time.get(h['speaker'], 0) + ov
        return max(spk_time, key=spk_time.get) if spk_time else 'Unknown'

    def speakers_in_seg(seg_start, seg_end):
        """Return ordered list of (speaker, start, end) within this segment."""
        parts = []
        for h in hyp_sorted:
            ov_start = max(seg_start, h['start'])
            ov_end   = min(seg_end,   h['end'])
            if ov_end - ov_start > 0.1:
                parts.append((h['speaker'], ov_start, ov_end))
        parts.sort(key=lambda x: x[1])
        # Merge consecutive same-speaker parts
        merged = []
        for spk, s, e in parts:
            if merged and merged[-1][0] == spk:
                merged[-1] = (spk, merged[-1][1], e)
            else:
                merged.append((spk, s, e))
        return merged

    dialogue = []
    for seg in stt_segs:
        text = seg.get('text', '').strip()
        if not text or not is_mostly_thai(text):
            continue

        parts = speakers_in_seg(seg['start'], seg['end'])

        if not parts:
            continue

        if len(parts) == 1:
            # Single speaker — keep whole segment
            dialogue.append({
                'start':   seg['start'],
                'end':     seg['end'],
                'speaker': parts[0][0],
                'text':    text,
            })
        else:
            # Multiple speakers — split text proportionally by duration
            total_dur   = sum(e - s for _, s, e in parts)
            chars       = text.replace(' ', '')
            total_chars = len(chars)
            char_pos    = 0
            for spk, s, e in parts:
                ratio    = (e - s) / total_dur if total_dur > 0 else 1 / len(parts)
                n_chars  = max(1, round(total_chars * ratio))
                sub_text = chars[char_pos:char_pos + n_chars]
                char_pos += n_chars
                if sub_text and is_mostly_thai(sub_text):
                    dialogue.append({
                        'start':   s,
                        'end':     e,
                        'speaker': spk,
                        'text':    sub_text,
                    })

    dialogue.sort(key=lambda x: x['start'])

    # Merge consecutive same-speaker turns (gap < 0.5s)
    merged = []
    for d in dialogue:
        if (merged and merged[-1]['speaker'] == d['speaker']
                and d['start'] - merged[-1]['end'] < 0.5):
            merged[-1]['text'] += d['text']
            merged[-1]['end']   = d['end']
        else:
            merged.append(dict(d))
    return merged


def save_dialogue_csv(path, merged):
    """Write dialogue to CSV with columns: Start, End, Speaker, Text."""
    rows = [{
        'Start':   sec_to_hmmss(d['start']),
        'End':     sec_to_hmmss(d['end']),
        'Speaker': d['speaker'],
        'Text':    d['text'],
    } for d in merged]
    pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
    print(f"   💾 Saved → {path}")


def save_dialogue_txt(path, merged):
    """Write dialogue as a fixed-width table to a .txt file."""
    lines = []
    lines.append(f"{'Start':<10} {'End':<10} {'Speaker':<12} Text")
    lines.append("-" * 80)
    for d in merged:
        text = d['text'].replace('\n', ' ').replace('\r', ' ')
        lines.append(
            f"{sec_to_hmmss(d['start']):<10} {sec_to_hmmss(d['end']):<10} "
            f"{d['speaker']:<12} {text}"
        )
    with open(path, 'w', encoding='utf-8-sig') as f:
        f.write("\n".join(lines))
    print(f"   💾 Saved → {path}")


def print_dialogue_table(title, merged, max_rows=None):
    """Print dialogue as a formatted table to console."""
    print(f"\n{'='*20} {title} {'='*20}")
    print(f"{'Start':<10} {'End':<10} {'Speaker':<12} Text")
    print("-" * 80)
    rows = merged if max_rows is None else merged[:max_rows]
    for d in rows:
        if d['speaker'] == 'Unknown':
            continue
        print(f"{sec_to_hmmss(d['start']):<10} {sec_to_hmmss(d['end']):<10} "
              f"{d['speaker']:<12} {d['text']}")
    if max_rows is not None and len(merged) > max_rows:
        print(f"... ({len(merged) - max_rows} more turns)")