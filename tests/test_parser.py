import pytest
from pathlib import Path
from piano_neuronal.s1_data.dataset_parser import parse_filename, discover_all_files, _FILENAME_RE


# Known filenames from the actual dataset
SAMPLE_FILENAMES = [
    "01-PedalOffForte1Close.flac",
    "01-PedalOffForte2Close.flac",
    "44-PedalOnPiano1Ambient.flac",
    "88-PedalOnC8MezzoPiano2Close.flac",  # hypothetical edge case
    "01-PedalOffPianissimo1Close.flac",
    "01-PedalOffMezzoForte1Ambient.flac",
]


class TestFilenameRegex:
    def test_basic_match(self):
        m = _FILENAME_RE.search("01-PedalOffForte1Close.flac")
        assert m is not None
        assert m.group(1) == "01"
        assert m.group(2) == "PedalOff"
        assert m.group(3) == "Forte"
        assert m.group(4) == "1"
        assert m.group(5) == "Close"

    def test_all_velocity_layers(self):
        for vel in ["Pianissimo", "Piano", "MezzoPiano", "MezzoForte", "Forte"]:
            filename = f"44-PedalOn{vel}1Close.flac"
            m = _FILENAME_RE.search(filename)
            assert m is not None, f"Failed to match velocity layer: {vel}"
            assert m.group(3) == vel

    def test_both_mics(self):
        for mic in ["Close", "Ambient"]:
            filename = f"44-PedalOffForte1{mic}.flac"
            m = _FILENAME_RE.search(filename)
            assert m is not None, f"Failed to match mic: {mic}"
            assert m.group(5) == mic

    def test_both_pedal_states(self):
        for pedal in ["PedalOn", "PedalOff"]:
            filename = f"44-{pedal}Forte1Close.flac"
            m = _FILENAME_RE.search(filename)
            assert m is not None, f"Failed to match pedal: {pedal}"

    def test_round_robins(self):
        for rr in ["1", "2"]:
            filename = f"44-PedalOffForte{rr}Close.flac"
            m = _FILENAME_RE.search(filename)
            assert m is not None, f"Failed to match round-robin: {rr}"

    def test_accepts_valid_stem(self):
        """Regex matches filename stems (extension is filtered by discover_all_files)."""
        m = _FILENAME_RE.search("01-PedalOffForte1Close")
        assert m is not None


class TestParseFilename:
    def test_note_01_is_A0(self):
        metadata = parse_filename(Path("01-PedalOffForte1Close.flac"))
        assert metadata["midi_note"] == 21
        assert metadata["note_name"] == "A0"

    def test_note_44_is_E4(self):
        # note_index 44 → MIDI 64 → E4
        metadata = parse_filename(Path("44-PedalOffForte1Close.flac"))
        assert metadata["midi_note"] == 64

    def test_note_88_is_C8(self):
        metadata = parse_filename(Path("88-PedalOffForte1Close.flac"))
        assert metadata["midi_note"] == 108
        assert metadata["note_name"] == "C8"

    def test_velocity_mapping(self):
        metadata = parse_filename(Path("44-PedalOffForte1Close.flac"))
        assert metadata["velocity_layer"] == "Forte"
        assert metadata["velocity_lovel"] == 102
        assert metadata["velocity_hivel"] == 127
        assert metadata["velocity_midi_center"] == 114
        assert abs(metadata["velocity_continuous"] - 114 / 127) < 1e-6

    def test_pp_velocity(self):
        metadata = parse_filename(Path("44-PedalOffPianissimo1Close.flac"))
        assert metadata["velocity_layer"] == "Pianissimo"
        assert metadata["velocity_lovel"] == 1
        assert metadata["velocity_hivel"] == 33

    def test_ambient_mic(self):
        metadata = parse_filename(Path("44-PedalOffForte1Ambient.flac"))
        assert metadata["mic"] == "Ambient"

    def test_pedal_on(self):
        metadata = parse_filename(Path("44-PedalOnForte1Close.flac"))
        assert metadata["pedal"] == "On"

    def test_round_robin(self):
        metadata = parse_filename(Path("44-PedalOffForte2Close.flac"))
        assert metadata["round_robin"] == 2