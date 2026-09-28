# Pause Annotator

A semi-automated workflow for annotating the internal structure of speech pauses. Silent pauses are rarely completely silent: they often contain breath noises, clicks and other non-verbal sounds. Pause Annotator detects pauses and the breaths and clicks inside them, and writes the results to Praat TextGrids, ready for manual checking and correction in Praat.

The tool is designed for close-microphone recordings with a homogeneous sound quality.

Sascha Schäfer & Jürgen Trouvain, Language Science and Technology, Saarland University

## Installation

You need Python 3.8 or newer with Tkinter, plus four packages:

```
pip install -r requirements.txt
```

Tkinter is included in the Python installers from python.org. If you use Homebrew Python on macOS, install it with `brew install python-tk`; on Debian or Ubuntu, use `sudo apt install python3-tk`.

## Starting the program

```
python pause_annotator.py
```

1. Choose the folder that contains your sound files (WAV, AIFF, FLAC, MP3 or OGG).
2. Choose which steps to run and adjust the settings if needed.
3. Click **Run**.

The results are written as TextGrids with the same names as the sound files to a subfolder called `annotated` inside the sound files folder. The original files are not changed. Files in subfolders are not processed. Stereo files are mixed to mono before the analysis.

Click **Stop** to stop after the current step; the file that is being processed is then not saved. **Restore defaults** resets all settings.

## The workflow

| Step | Tier | Content |
|---|---|---|
| 1 | PauseDetect | IPUs (`ipu`) and pauses (`pause`) |
| 2 | BreathDetect | Breath noises and silence inside each pause; IPUs are marked `ipu` |
| 2 | AlignTier | A copy of BreathDetect for manual correction |
| 3 | ClickDetect | Point tier with one point per click, inside the pauses |
| 4 | – | Manual correction in Praat |

Step 4 is done by human annotators after the automatic steps: they adjust pause boundaries in PauseDetect and AlignTier, boundaries between breath and silence in AlignTier, and click positions in ClickDetect. BreathDetect keeps the unchanged automatic result, so the corrected AlignTier can be compared with it.

### Working with your own pause segmentation

If your TextGrids already contain a division into pauses and IPUs, switch Step 1 off and run only Steps 2 and/or 3. In the **Existing segmentation** box on the Step 1 tab, enter:

* **Pause tier name**: the name of the interval tier with your segmentation.
* **Pause label**: the label of your pause intervals, for example `pause`, `sil`, `<p:>` or `#`. Leave the field empty if your pauses are empty intervals. Upper and lower case are not distinguished.
* **TextGrid folder**: where your TextGrids are, if not in the sound files folder. Each TextGrid must have the same name as its sound file.

Every interval that does not carry the pause label counts as an IPU, whatever its label. The new tiers are added to your TextGrid; your existing tiers are kept. If a TextGrid already has a tier with the name of a new tier (for example from an earlier run), that tier is replaced.

## Step 1: Pause detection

The sound is segmented with Praat's silences annotator (*Sound: To TextGrid (silences)*): intensity is computed, frames more than a threshold below the loudest frame count as silent, and short silent and sounding stretches are then removed. Three optional stages refine the result. With all three switched off, the output is identical to Praat's silences annotator with the same settings.

1. **Band-pass filter**: removes low-frequency rumble and high-frequency hiss before the intensity analysis.
2. **Burst correction**: short bursts of energy right before speech onset (typically clicks) would otherwise be counted as the start of the IPU. They are moved into the preceding pause, so that Step 3 can find them there. A burst qualifies if it is shorter than the maximum burst duration, lies within the search window after the IPU onset, and is followed by a clear drop in energy.
3. **Voicing check**: intensity alone cannot tell speech from other loud events. Inhalations, coughs, lip smacks or microphone noise lasting longer than the minimum IPU duration become IPUs, although they belong to the pause. The voicing check relabels every IPU with less than the minimum voiced time as a pause. The pitch tracker is set up very liberally, so that quiet or creaky speech still counts as voiced. Note that voiceless speech (whispering, a voiceless "shh" or "psst") will also be relabelled as pause, while voiced non-speech (voiced laughter, humming) stays an IPU.

| Setting | Default | Meaning |
|---|---|---|
| Silence threshold | 30 dB | Frames more than this below the loudest frame are silent |
| Minimum pause duration | 0.2 s | Shorter silences become part of the IPU |
| Minimum IPU duration | 0.1 s | Shorter sounding stretches become part of the pause |
| Band-pass low / high cut-off | 80 / 8000 Hz | Pass band of the filter |
| Intensity minimum pitch | 100 Hz | Determines the analysis window of the intensity |
| Intensity time step | 0.008 s | 0 lets Praat choose (0.8 / minimum pitch) |
| Maximum burst duration | 0.06 s | Longer bursts are not moved |
| Search window after IPU onset | 0.3 s | Bursts later than this are not moved |
| Window for energy drop | 0.03 s | Time after the burst in which energy must drop |
| Energy drop: dB above threshold | 4 dB | How far the energy must drop |
| Minimum voiced time per IPU | 0.05 s | Total voiced time needed to keep an IPU |
| Pitch floor / ceiling | 50 / 700 Hz | Pitch range of the voicing check |
| Pitch time step | 0.01 s | |
| Silence / voicing threshold | 0.01 / 0.2 | Praat's pitch settings (lower = more liberal) |
| Octave / octave-jump / voiced-unvoiced cost | 0.01 / 0.2 / 0.1 | Praat's pitch settings |

## Step 2: Breath detection

The sound is filtered with a Hann band-pass filter (Praat's *Filter (pass Hann band)*) to isolate the frequency range of breath noise, and its intensity is computed. Inside each pause, the intensity is read in fixed steps. Stretches at or above the threshold that last at least the minimum breath duration are labelled as breath; the rest of the pause is labelled as silence.

In the **absolute** threshold mode, the threshold is a fixed intensity value. It depends on the recording level, so it may need adjusting for recordings made with a different gain. In the **relative** mode, the threshold is set a number of dB below the loudest point of the filtered file, which makes it independent of the recording level. The relative default of 50 dB is a starting point and should be checked on your data.

| Setting | Default | Meaning |
|---|---|---|
| Threshold mode | absolute | `absolute` or `relative` |
| Absolute threshold | 20 dB | Used in absolute mode |
| Relative threshold | 50 dB | dB below the file peak, used in relative mode |
| Minimum breath duration | 0.2 s | Shorter stretches are labelled as silence |
| Breath / silence label | `breath` / `sil` | Labels in BreathDetect and AlignTier |
| Band-pass low / high cut-off | 300 / 3000 Hz | Pass band of the Hann filter |
| Hann smoothing | 100 Hz | Width of the filter slopes |
| Intensity minimum pitch | 100 Hz | Determines the analysis window of the intensity |
| Analysis step | 0.01 s | Step in which the intensity is read |
| Bridge gaps shorter than | 0 s (off) | Joins breath stretches separated by a short dip |

## Step 3: Click detection

The sound is filtered with a Butterworth band-pass filter in a high frequency range, and its energy is measured in very short frames. The threshold is a fraction of the loudest frame in the whole file. Inside each pause, short stretches above the threshold whose duration lies between the minimum and maximum click duration are clicks. Bursts that are very close together are merged, and clicks within the clustering window are marked by a single point at the loudest click.

Because the reference is the loudest frame in the file, which is usually a sibilant in the speech, speakers with very strong sibilants will get fewer clicks detected. Lower the threshold fraction if clicks are missed.

| Setting | Default | Meaning |
|---|---|---|
| Threshold | 0.15 | Fraction of the RMS of the loudest frame in the file |
| Minimum / maximum click duration | 0.5 / 10 ms | Allowed click duration |
| Click label | `click` | Label of the points |
| Band-pass low / high cut-off | 3000 / 10000 Hz | Capped at the Nyquist frequency |
| Butterworth filter order | 4 | |
| Frame length / hop | 1 / 0.5 ms | Energy frames |
| Merge bursts closer than | 2 ms | |
| Cluster clicks within | 30 ms | One point per cluster, at the loudest click |

## Notes

* Very long files are filtered in 60-second chunks, so memory use stays moderate. The result is practically identical to filtering the whole file at once.
* If a file cannot be processed (for example because its TextGrid or pause tier is missing), it is skipped and the reason is shown in the log window. The remaining files are processed as usual.
* Pause Annotator uses the version of Praat that is built into Parselmouth.

## Software used

Praat (Boersma & Weenink), Parselmouth (Jadoul, Thompson & de Boer 2018), NumPy (Harris et al. 2020), SciPy (Virtanen et al. 2020) and TextGridTools (Buschmeier & Włodarczak 2013).
