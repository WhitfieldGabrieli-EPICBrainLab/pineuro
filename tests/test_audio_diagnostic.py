"""
Audio diagnostic tests for Self/Other voice localizer task.

Isolates each layer of the audio pipeline to pinpoint why audio clips
aren't playing during the self/other task.

Usage:
    python tests/test_audio_diagnostic.py              # Run all tests
    python tests/test_audio_diagnostic.py --test 1     # Run specific test
    python tests/test_audio_diagnostic.py --test 1,3,5 # Run multiple tests
"""

import argparse
import csv
import os
import sys
import time
import wave
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust these if your data lives elsewhere
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # pineuro repo root
DATA_DIR = PROJECT_ROOT / "data" / "R33"
SELF_WAV_DIR = DATA_DIR / "self"
OTHER_WAV_DIR = DATA_DIR / "other"
STIM_FILE = (
    DATA_DIR
    / "task_stimulus"
    / "sub-R33rtsz036_ses-localizer_task-selfother_run-01.txt"
)


def _pick_wav(directory: Path) -> Path | None:
    """Return first .wav file found in *directory*, or None."""
    wavs = sorted(directory.glob("*.wav"))
    return wavs[0] if wavs else None


def _banner(test_num: int, title: str):
    print(f"\n{'='*60}")
    print(f"  TEST {test_num}: {title}")
    print(f"{'='*60}")


# ===================================================================
# Test 1 — Audio backend detection
# ===================================================================
def test_1_backend_detection():
    _banner(1, "Audio backend detection")

    # PsychoPy prefs
    try:
        from psychopy import prefs

        audio_lib = prefs.hardware.get("audioLib", ["<not set>"])
        print(f"  PsychoPy audioLib pref : {audio_lib}")
    except Exception as e:
        print(f"  Could not read PsychoPy prefs: {e}")

    # pygame availability
    try:
        import pygame

        pygame.mixer.init()
        freq, size, channels = pygame.mixer.get_init()
        print(f"  pygame.mixer           : freq={freq}, size={size}, channels={channels}")
        pygame.mixer.quit()
    except Exception as e:
        print(f"  pygame.mixer           : UNAVAILABLE ({e})")

    # sounddevice availability
    try:
        import sounddevice as sd

        devs = sd.query_devices()
        default_out = sd.default.device[1]
        print(f"  sounddevice            : {len(devs)} devices, default output={default_out}")
        if default_out is not None and default_out >= 0:
            print(f"    default output device: {devs[default_out]['name']}")
    except Exception as e:
        print(f"  sounddevice            : UNAVAILABLE ({e})")

    # PTB availability
    try:
        import psychtoolbox as ptb  # noqa: F401

        print(f"  psychtoolbox (PTB)     : available")
    except Exception as e:
        print(f"  psychtoolbox (PTB)     : UNAVAILABLE ({e})")

    print("\n  [INFO] PsychoPy tries backends in audioLib order. First working one wins.")
    print("  RESULT: CHECK — review the output above for your active backend.")


# ===================================================================
# Test 2 — Raw WAV file validation
# ===================================================================
def test_2_wav_validation():
    _banner(2, "Raw WAV file validation")

    for label, wav_dir in [("self", SELF_WAV_DIR), ("other", OTHER_WAV_DIR)]:
        wav_path = _pick_wav(wav_dir)
        if wav_path is None:
            print(f"  [{label}] NO WAV FILES FOUND in {wav_dir}")
            print(f"  RESULT: FAIL")
            continue

        print(f"  [{label}] File: {wav_path.name}")

        # stdlib wave
        try:
            with wave.open(str(wav_path), "rb") as wf:
                sr = wf.getframerate()
                ch = wf.getnchannels()
                sw = wf.getsampwidth()
                nf = wf.getnframes()
                dur = nf / sr
                print(f"    wave module  : sr={sr}, channels={ch}, sample_width={sw}, "
                      f"frames={nf}, duration={dur:.3f}s")
        except Exception as e:
            print(f"    wave module  : FAILED ({e})")

        # scipy check for non-zero data
        try:
            from scipy.io import wavfile

            sr2, data = wavfile.read(str(wav_path))
            peak = abs(data).max()
            print(f"    scipy        : sr={sr2}, shape={data.shape}, peak={peak}")
            if peak == 0:
                print(f"    WARNING      : Audio data is all zeros!")
            else:
                print(f"    Data         : OK (non-silent)")
        except Exception as e:
            print(f"    scipy        : FAILED ({e})")

    print("\n  RESULT: PASS if both files have non-zero data and valid headers.")


# ===================================================================
# Test 3 — Direct pygame playback (bypass PsychoPy)
# ===================================================================
def test_3_pygame_direct():
    _banner(3, "Direct pygame playback (bypass PsychoPy)")

    wav_path = _pick_wav(SELF_WAV_DIR)
    if wav_path is None:
        print(f"  No WAV files in {SELF_WAV_DIR}")
        print("  RESULT: SKIP")
        return

    try:
        import pygame

        pygame.mixer.init()
        snd = pygame.mixer.Sound(str(wav_path))
        duration = snd.get_length()
        print(f"  File    : {wav_path.name}")
        print(f"  Duration: {duration:.3f}s")
        print(f"  Playing via pygame.mixer.Sound ...")
        snd.play()
        time.sleep(duration + 0.5)
        pygame.mixer.quit()
        print(f"  Playback finished.")
        print(f"  RESULT: PASS if you heard audio. FAIL if silent.")
    except Exception as e:
        print(f"  pygame playback FAILED: {e}")
        print(f"  RESULT: FAIL — OS-level or pygame audio issue")


# ===================================================================
# Test 4 — PsychoPy Sound basic playback (no secs param)
# ===================================================================
def test_4_psychopy_basic():
    _banner(4, "PsychoPy Sound — basic playback (no secs param)")

    wav_path = _pick_wav(SELF_WAV_DIR)
    if wav_path is None:
        print(f"  No WAV files in {SELF_WAV_DIR}")
        print("  RESULT: SKIP")
        return

    try:
        from psychopy import sound

        print(f"  File   : {wav_path.name}")
        print(f"  Creating sound.Sound(path) — NO secs parameter ...")
        snd = sound.Sound(str(wav_path))
        dur = snd.getDuration() or 2.0
        print(f"  Duration reported: {dur:.3f}s")
        print(f"  Playing ...")
        snd.play()
        time.sleep(dur + 0.5)
        snd.stop()
        print(f"  Playback finished.")
        print(f"  RESULT: PASS if you heard audio. FAIL if silent.")
    except Exception as e:
        print(f"  PsychoPy Sound FAILED: {e}")
        import traceback; traceback.print_exc()
        print(f"  RESULT: FAIL")


# ===================================================================
# Test 5 — PsychoPy Sound with secs parameter (matches SelfOtherTask)
# ===================================================================
def test_5_psychopy_with_secs():
    _banner(5, "PsychoPy Sound — with secs param (SelfOtherTask pattern)")

    wav_path = _pick_wav(SELF_WAV_DIR)
    if wav_path is None:
        print(f"  No WAV files in {SELF_WAV_DIR}")
        print("  RESULT: SKIP")
        return

    # Get actual duration from wave module
    with wave.open(str(wav_path), "rb") as wf:
        actual_dur = wf.getnframes() / wf.getframerate()

    try:
        from psychopy import sound

        print(f"  File        : {wav_path.name}")
        print(f"  Actual dur  : {actual_dur:.3f}s")
        print(f"  Creating sound.Sound(path, secs={actual_dur:.6f}) — WITH secs parameter ...")
        snd = sound.Sound(str(wav_path), secs=actual_dur)
        snd.setVolume(1.0)
        reported_dur = snd.getDuration() or actual_dur
        print(f"  Reported dur: {reported_dur:.3f}s")
        print(f"  Playing ...")
        snd.play()
        time.sleep(actual_dur + 0.5)
        snd.stop()
        print(f"  Playback finished.")
        print(f"  RESULT: PASS if you heard audio. FAIL if silent.")
        print(f"  Compare to Test 4 — if Test 4 played but this didn't, "
              f"the secs parameter is the problem.")
    except Exception as e:
        print(f"  PsychoPy Sound with secs FAILED: {e}")
        import traceback; traceback.print_exc()
        print(f"  RESULT: FAIL")


# ===================================================================
# Test 6 — Pre-load and sequential replay
# ===================================================================
def test_6_preload_replay():
    _banner(6, "Pre-load and sequential replay (simulate one voice block)")

    wavs = sorted(SELF_WAV_DIR.glob("*.wav"))[:5]
    if len(wavs) < 5:
        print(f"  Need 5 WAV files in {SELF_WAV_DIR}, found {len(wavs)}")
        print("  RESULT: SKIP")
        return

    try:
        from psychopy import sound

        # Pre-load (matching SelfOtherTask.setup pattern)
        preloaded = {}
        print("  Pre-loading 5 sounds with secs parameter ...")
        for wav_path in wavs:
            with wave.open(str(wav_path), "rb") as wf:
                dur = wf.getnframes() / wf.getframerate()
            snd = sound.Sound(str(wav_path), secs=dur)
            snd.setVolume(1.0)
            preloaded[str(wav_path)] = (snd, dur)
            print(f"    Loaded: {wav_path.name} (dur={dur:.3f}s)")

        # First pass playback
        print("\n  --- First playback pass ---")
        for wav_path in wavs:
            snd, dur = preloaded[str(wav_path)]
            print(f"    Playing {wav_path.name} ...", end=" ", flush=True)
            snd.play()
            time.sleep(dur)
            snd.stop()
            print("done")
            time.sleep(0.3)  # ISI

        # Second pass (re-playability test)
        print("\n  --- Second playback pass (re-playability) ---")
        for wav_path in wavs:
            snd, dur = preloaded[str(wav_path)]
            print(f"    Replaying {wav_path.name} ...", end=" ", flush=True)
            snd.play()
            time.sleep(dur)
            snd.stop()
            print("done")
            time.sleep(0.3)

        print(f"\n  RESULT: PASS if all 10 plays were audible.")
        print(f"  If first pass played but second didn't, re-playability bug.")
    except Exception as e:
        print(f"  Pre-load/replay FAILED: {e}")
        import traceback; traceback.print_exc()
        print(f"  RESULT: FAIL")


# ===================================================================
# Test 7 — Stim file path resolution
# ===================================================================
def test_7_stim_file_paths():
    _banner(7, "Stim file path resolution & loading")

    if not STIM_FILE.exists():
        print(f"  Stim file not found: {STIM_FILE}")
        print("  RESULT: SKIP")
        return

    print(f"  Stim file: {STIM_FILE.name}")

    rows = []
    with open(STIM_FILE) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            while len(row) < 4:
                row.append("")
            rows.append(row)

    print(f"  Total rows: {len(rows)}")

    missing = []
    loadable = 0
    load_errors = []

    from psychopy import sound

    for i, row in enumerate(rows):
        audio_path = row[0]
        duration = float(row[1])
        exists = Path(audio_path).exists()

        if not exists:
            missing.append((i, audio_path))
            continue

        # Try loading as PsychoPy Sound (with secs, matching SelfOtherTask)
        try:
            snd = sound.Sound(audio_path, secs=duration)
            snd.setVolume(1.0)
            reported_dur = snd.getDuration()
            loadable += 1
            status = "OK"
            if reported_dur is not None and abs(reported_dur - duration) > 0.5:
                status = f"OK (dur mismatch: reported={reported_dur:.3f} vs stim={duration:.3f})"
            print(f"    Row {i+1:2d}: {Path(audio_path).name:20s} dur={duration:.3f}s  {status}")
        except Exception as e:
            load_errors.append((i, audio_path, str(e)))
            print(f"    Row {i+1:2d}: {Path(audio_path).name:20s} LOAD ERROR: {e}")

    print(f"\n  Summary:")
    print(f"    Total audio rows : {len(rows)}")
    print(f"    Files exist      : {len(rows) - len(missing)}")
    print(f"    Missing files    : {len(missing)}")
    print(f"    Loadable sounds  : {loadable}")
    print(f"    Load errors      : {len(load_errors)}")

    if missing:
        print(f"\n  MISSING FILES:")
        for idx, path in missing:
            print(f"    Row {idx+1}: {path}")

    if load_errors:
        print(f"\n  LOAD ERRORS:")
        for idx, path, err in load_errors:
            print(f"    Row {idx+1}: {Path(path).name} — {err}")

    if not missing and not load_errors:
        print(f"\n  RESULT: PASS — all {len(rows)} audio files exist and load successfully.")
    else:
        print(f"\n  RESULT: FAIL — see issues above.")


# ===================================================================
# Test 8 — PsychoPy Sound with pygame backend forced
# ===================================================================
def test_8_psychopy_pygame_backend():
    _banner(8, "PsychoPy Sound with pygame backend (fix verification)")

    wav_path = _pick_wav(SELF_WAV_DIR)
    if wav_path is None:
        print(f"  No WAV files in {SELF_WAV_DIR}")
        print("  RESULT: SKIP")
        return

    try:
        from psychopy import prefs
        prefs.hardware['audioLib'] = ['pygame']

        # Force re-import of sound module with new prefs
        import importlib
        from psychopy import sound
        importlib.reload(sound)

        print(f"  audioLib forced to: {prefs.hardware['audioLib']}")
        print(f"  File: {wav_path.name}")
        print(f"  Creating sound.Sound(path) with pygame backend ...")
        snd = sound.Sound(str(wav_path))
        dur = snd.getDuration() or 2.0
        print(f"  Duration reported: {dur:.3f}s")
        print(f"  Playing ...")
        snd.play()
        time.sleep(dur + 0.5)
        snd.stop()
        print(f"  Playback finished.")
        print(f"  RESULT: PASS if you heard audio. FAIL if silent.")
    except Exception as e:
        print(f"  PsychoPy+pygame FAILED: {e}")
        import traceback; traceback.print_exc()
        print(f"  RESULT: FAIL")


# ===================================================================
# Main
# ===================================================================
ALL_TESTS = {
    1: test_1_backend_detection,
    2: test_2_wav_validation,
    3: test_3_pygame_direct,
    4: test_4_psychopy_basic,
    5: test_5_psychopy_with_secs,
    6: test_6_preload_replay,
    7: test_7_stim_file_paths,
    8: test_8_psychopy_pygame_backend,
}


def main():
    parser = argparse.ArgumentParser(description="Audio diagnostic tests for self/other task")
    parser.add_argument(
        "--test", "-t",
        type=str,
        default=None,
        help="Comma-separated test numbers to run (e.g. '1,3,5'). Default: all.",
    )
    args = parser.parse_args()

    if args.test:
        test_nums = [int(x.strip()) for x in args.test.split(",")]
    else:
        test_nums = list(ALL_TESTS.keys())

    print("Audio Diagnostic Tests for Self/Other Task")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Data dir     : {DATA_DIR}")
    print(f"Platform     : {sys.platform}")
    print(f"Python       : {sys.version}")

    for num in test_nums:
        if num not in ALL_TESTS:
            print(f"\nUnknown test number: {num}")
            continue
        try:
            ALL_TESTS[num]()
        except Exception as e:
            print(f"\n  UNEXPECTED ERROR in test {num}: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*60}")
    print("  DONE — Review results above.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
