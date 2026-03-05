# -*- coding: utf-8 -*-
"""
Sandhi Engine
==============
Wrappers for Sanskrit sandhi operations:

* **Sandhi joining** — Uses the Perl scripts in ``sandhi/`` via subprocess.
  Input is transliterated IAST → WX before calling Perl, and the result is
  converted WX → IAST.
* **Sandhi splitting** — Uses the neural sandhi splitter in ``sandhi_split/``
  (TensorFlow SavedModel).  Runs entirely offline.

Both operations are fully offline and cross-platform (Perl must be installed
for sandhi joining; TensorFlow 1.x for sandhi splitting).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from engines import transliteration_engine as te

logger = logging.getLogger("varnabuddhi.sandhi")


class SandhiJoinError(RuntimeError):
    """Raised when sandhi joining fails."""


class SandhiSplitError(RuntimeError):
    """Raised when sandhi splitting fails."""


# Common Perl install paths to check when ``perl`` is not on PATH
_PERL_FALLBACK_PATHS = [
    r"C:\Strawberry\perl\bin\perl.exe",  # Strawberry Perl (Windows)
    r"C:\Perl64\bin\perl.exe",           # ActivePerl 64-bit
    r"C:\Perl\bin\perl.exe",             # ActivePerl 32-bit
    "/usr/bin/perl",                      # Linux / macOS
    "/usr/local/bin/perl",               # Homebrew etc.
]


def _find_perl() -> Optional[str]:
    """Locate a usable ``perl`` executable."""
    found = shutil.which("perl")
    if found:
        return found
    for candidate in _PERL_FALLBACK_PATHS:
        if os.path.isfile(candidate):
            return candidate
    return None


class SandhiEngine:
    """Perform sandhi joining and splitting operations."""

    def __init__(
        self,
        sandhi_dir: str | Path,
        sandhi_split_dir: str | Path,
    ) -> None:
        self.sandhi_dir = Path(sandhi_dir)
        self.sandhi_split_dir = Path(sandhi_split_dir)
        self._perl_path: Optional[str] = _find_perl()
        self._split_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Sandhi joining (Perl-based)
    # ------------------------------------------------------------------
    @property
    def join_available(self) -> bool:
        """Return True if the Perl-based sandhi join engine is usable."""
        return (
            self._perl_path is not None
            and (self.sandhi_dir / "any_sandhi.pl").exists()
            and (self.sandhi_dir / "sandhi.pl").exists()
        )

    def join(self, left: str, right: str) -> str:
        """Apply padānta sandhi to join *left* and *right* words.

        Parameters
        ----------
        left, right : str
            Sanskrit words in IAST.  They will be transliterated to WX
            internally for the Perl engine.

        Returns
        -------
        str
            The joined result in IAST.
        """
        if not self.join_available:
            raise SandhiJoinError(
                "Sandhi joining requires Perl and the sandhi/ scripts.  "
                "Ensure Perl is installed and sandhi/any_sandhi.pl exists."
            )

        wx_left = te.to_wx(left)
        wx_right = te.to_wx(right)

        # Build a small Perl driver that loads the sandhi library and calls it.
        # The apavAxa_any.pl script tries to require "../paths.pl" and uses
        # $GlblVar::LTPROCBIN (a morphological analyser binary) for a few
        # edge-case rules.  We stub those out so the hundreds of
        # pattern-matching sandhi rules still work.
        sandhi_posix = self.sandhi_dir.as_posix()
        driver = (
            # Stub the GlblVar namespace so require "../paths.pl" becomes a no-op
            f'package GlblVar; '
            f'our $CGIDIR = ""; our $SCL_CGI = ""; our $LTPROCBIN = ""; '
            f'package main;\n'
            # Override require to silently skip paths.pl
            f'BEGIN {{ '
            f'  my $orig = \\&CORE::GLOBAL::require; '
            f'  *CORE::GLOBAL::require = sub {{ '
            f'    return 1 if $_[0] =~ /paths\.pl/; '
            f'    CORE::require($_[0]); '
            f'  }}; '
            f'}}\n'
            f'use lib "{sandhi_posix}";\n'
            f'do "{sandhi_posix}/apavAxa_any.pl" or 1;\n'
            f'do "{sandhi_posix}/any_sandhi.pl" or 1;\n'
            f'do "{sandhi_posix}/sandhi.pl" or 1;\n'
            f'my $result = sandhi("{wx_left}", "{wx_right}");\n'
            f'print $result;\n'
        )

        try:
            proc = subprocess.run(
                [self._perl_path, "-e", driver],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.sandhi_dir),
            )
        except FileNotFoundError:
            raise SandhiJoinError("Perl executable not found.")
        except subprocess.TimeoutExpired:
            raise SandhiJoinError("Sandhi join timed out.")

        if proc.returncode != 0:
            logger.warning("Perl sandhi stderr: %s", proc.stderr.strip())
            raise SandhiJoinError(f"Perl sandhi exited with code {proc.returncode}")

        raw_output = proc.stdout.strip()
        if not raw_output:
            logger.warning("Empty sandhi output for '%s' + '%s'", left, right)
            return left + right

        # The output format is: ":result1:result2,...,sandhiName,..."
        # Take the first result token before the first comma
        parts = raw_output.split(",")
        candidates = parts[0] if parts else raw_output
        # Candidates are colon-separated alternatives; take the first non-empty
        for candidate in candidates.split(":"):
            candidate = candidate.strip()
            if candidate:
                return te.from_wx(candidate)

        return left + right

    def join_words(self, words: List[str]) -> str:
        """Sequentially apply sandhi joining to a list of IAST words.

        .. note:: This fuses ALL words into one string, which is rarely
           correct for a full sentence.  Prefer :meth:`join_pairwise` for
           displaying sandhi at word boundaries.
        """
        if not words:
            return ""
        result = words[0]
        for w in words[1:]:
            try:
                result = self.join(result, w)
            except SandhiJoinError as exc:
                logger.warning("Sandhi join failed for '%s' + '%s': %s", result, w, exc)
                result = result + " " + w  # Fallback: space-separated
        return result

    def join_pairwise(
        self, words: List[str]
    ) -> Tuple[str, List[Tuple[str, str, str]]]:
        """Apply sandhi between each pair of adjacent words independently.

        Returns
        -------
        (sandhied_sentence, details)
            *sandhied_sentence* is a space-joined string where each
            word-boundary has external sandhi applied.  *details* is a
            list of (left, right, joined) tuples for every boundary
            where the result differed from simple concatenation.
        """
        if len(words) < 2:
            return (" ".join(words), [])

        details: List[Tuple[str, str, str]] = []
        output_tokens: List[str] = []

        for i in range(len(words) - 1):
            left = words[i]
            right = words[i + 1]
            try:
                joined = self.join(left, right)
            except SandhiJoinError:
                joined = left + right  # fallback: just concatenate

            if joined != left + right and joined != left + " " + right:
                details.append((left, right, joined))

            # Heuristic: split the joined result to recover the modified
            # left-half and right-half so we can chain properly.
            # If the join produced a single token (no space), the boundary
            # is fused — store the left portion and let the right word be
            # re-evaluated on the next pair.
            if " " in joined:
                # Perl returned two tokens (e.g. visarga → "rāmo gacchati")
                parts = joined.split(None, 1)
                if i == 0:
                    output_tokens.append(parts[0])
                else:
                    # Replace the last token with the sandhi-modified version
                    output_tokens[-1] = parts[0]
                # Update the right word for the next iteration
                words[i + 1] = parts[1] if len(parts) > 1 else right
            else:
                # Fully fused (e.g. "tat" + "āha" → "tadāha") — treat as
                # a single token replacing both; skip right word.
                if i == 0:
                    output_tokens.append(joined)
                else:
                    output_tokens[-1] = joined
                # Mark right word as consumed so it isn't duplicated
                words[i + 1] = ""  # will be skipped or overwritten

        # Append the (possibly modified) last word if it wasn't consumed
        if words[-1]:
            output_tokens.append(words[-1])

        return (" ".join(t for t in output_tokens if t), details)

    # ------------------------------------------------------------------
    # Sandhi splitting (Python / TF-based)
    # ------------------------------------------------------------------
    @property
    def split_available(self) -> bool:
        """Return True if the neural sandhi splitter is usable."""
        if self._split_available is not None:
            return self._split_available
        self._split_available = (
            (self.sandhi_split_dir / "baba_sandhi_split.py").exists()
        )
        return self._split_available

    def split(self, text: str) -> str:
        """Split a sandhied Sanskrit text into its component words.

        Uses the neural sandhi splitter in ``sandhi_split/``.

        Parameters
        ----------
        text : str
            Sanskrit text in IAST (or Devanagari — auto-detected).

        Returns
        -------
        str
            The unsandhied (padapāṭha) form.
        """
        if not self.split_available:
            raise SandhiSplitError(
                "Sandhi splitter not available.  Ensure sandhi_split/ directory "
                "exists with baba_sandhi_split.py."
            )

        # Write input to a temp file, invoke the splitter, read output
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", encoding="utf-8", delete=False
            ) as tmp_in:
                tmp_in.write(text + "\n")
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path.replace(".txt", "_unsandhied.txt")

            proc = subprocess.run(
                [
                    sys.executable,
                    str(self.sandhi_split_dir / "baba_sandhi_split.py"),
                    "--input", tmp_in_path,
                    "--output", tmp_out_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.sandhi_split_dir),
            )

            if proc.returncode != 0:
                stderr = proc.stderr.strip()
                logger.warning("Sandhi split stderr: %s", stderr)
                # Provide a graceful fallback
                if "tensorflow" in stderr.lower() or "ModuleNotFoundError" in stderr:
                    raise SandhiSplitError(
                        "Sandhi splitting requires TensorFlow 1.x.  "
                        "Install with: pip install tensorflow==1.15.5"
                    )
                raise SandhiSplitError(f"Sandhi split exited with code {proc.returncode}")

            if os.path.exists(tmp_out_path):
                result = Path(tmp_out_path).read_text(encoding="utf-8").strip()
                return result
            else:
                logger.warning("Sandhi split output file not created.")
                return text

        except subprocess.TimeoutExpired:
            raise SandhiSplitError("Sandhi split timed out (120s).")
        finally:
            # Clean up temp files
            for p in (tmp_in_path, tmp_out_path):
                try:
                    if os.path.exists(p):
                        os.unlink(p)
                except OSError:
                    pass
