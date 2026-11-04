"""
Tests for BaseMeta, SDSSMeta, and PAQSMeta metadata classes.

Covers:
- Basic instantiation and property access
- Pydantic validation (RA/DEC bounds, title constraints, etc.)
- JSON serialization and deserialization
"""
import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from quasar_utils.loading.metadata import BaseMeta, PAQSMeta, SDSSMeta

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fits_file(tmp_path: Path) -> Path:
    """Create a temporary FITS file for testing."""
    fits_path = tmp_path / "test_spectrum.fits"
    fits_path.touch()
    return fits_path


@pytest.fixture
def valid_base_meta_dict(fits_file: Path) -> dict:
    """Create valid metadata dictionary for BaseMeta."""
    return {
        "path": str(fits_file),
        "title": "Test Spectrum",
        "ra": 180.0,
        "dec": 0.0,
        "z": 0.5,
        "luminosity_distance_mpc": 2000.0,
    }


@pytest.fixture
def valid_sdss_meta_dict(valid_base_meta_dict: dict) -> dict:
    """Create valid metadata dictionary for SDSSMeta."""
    return {
        **valid_base_meta_dict,
        "plate": 1234,
        "fiber": 567,
        "mjd": 55000,
    }


@pytest.fixture
def valid_paqs_meta_dict(valid_base_meta_dict: dict) -> dict:
    """Create valid metadata dictionary for PAQSMeta."""
    return {
        **valid_base_meta_dict,
        "obj_nme": "NGC_1234",
        "obj_uid": 999,
        "specuid": 777,
    }


# ============================================================================
# BaseMeta Tests: Basic Instantiation
# ============================================================================

class TestBaseMetaInstantiation:
    """Test basic instantiation of BaseMeta."""

    def test_create_base_meta_with_all_fields(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test creating BaseMeta with all required fields."""
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.title == "Test Spectrum"
        assert meta.ra == 180.0
        assert meta.dec == 0.0
        assert meta.z == 0.5
        assert meta.luminosity_distance_mpc == 2000.0

    def test_base_meta_path_property(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that path property returns correct value."""
        meta = BaseMeta(**valid_base_meta_dict)
        assert str(meta.path).endswith("test_spectrum.fits")

    def test_base_meta_is_frozen(self, valid_base_meta_dict: dict) -> None:
        """Test that BaseMeta is immutable (frozen)."""
        meta = BaseMeta(**valid_base_meta_dict)
        with pytest.raises(ValueError, match="frozen"):
            meta.title = "Modified Title"

    def test_base_meta_zero_values_allowed(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that zero values are allowed for non-negative fields."""
        valid_base_meta_dict["z"] = 0.0
        valid_base_meta_dict["luminosity_distance_mpc"] = 0.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.z == 0.0
        assert meta.luminosity_distance_mpc == 0.0


# ============================================================================
# Pydantic Validation: RA Field
# ============================================================================

class TestBaseMetaRAValidation:
    """Test RA field validation (0 <= RA < 360)."""

    def test_ra_at_minimum_boundary(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test RA at lower bound (0 degrees)."""
        valid_base_meta_dict["ra"] = 0.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.ra == 0.0

    def test_ra_at_maximum_boundary(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test RA near upper bound (359.999 degrees)."""
        valid_base_meta_dict["ra"] = 359.999
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.ra == 359.999

    def test_ra_below_minimum_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that RA < 0 raises ValidationError."""
        valid_base_meta_dict["ra"] = -0.1
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_ra_at_or_above_maximum_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that RA >= 360 raises ValidationError."""
        valid_base_meta_dict["ra"] = 360.0
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "less than 360" in str(exc_info.value)

    def test_ra_well_above_maximum_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that RA >> 360 raises ValidationError."""
        valid_base_meta_dict["ra"] = 500.0
        with pytest.raises(ValidationError):
            BaseMeta(**valid_base_meta_dict)


# ============================================================================
# Pydantic Validation: DEC Field
# ============================================================================

class TestBaseMetaDECValidation:
    """Test DEC field validation (-90 <= DEC <= 90)."""

    def test_dec_at_minimum_boundary(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test DEC at lower bound (-90 degrees)."""
        valid_base_meta_dict["dec"] = -90.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.dec == -90.0

    def test_dec_at_maximum_boundary(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test DEC at upper bound (90 degrees)."""
        valid_base_meta_dict["dec"] = 90.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.dec == 90.0

    def test_dec_below_minimum_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that DEC < -90 raises ValidationError."""
        valid_base_meta_dict["dec"] = -90.1
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "greater than or equal to -90" in str(exc_info.value)

    def test_dec_above_maximum_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that DEC > 90 raises ValidationError."""
        valid_base_meta_dict["dec"] = 90.1
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "less than or equal to 90" in str(exc_info.value)

    def test_dec_equatorial_region(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test DEC values near equator."""
        for dec_value in [-0.5, 0.0, 0.5]:
            valid_base_meta_dict["dec"] = dec_value
            meta = BaseMeta(**valid_base_meta_dict)
            assert meta.dec == dec_value


# ============================================================================
# Pydantic Validation: Title Field
# ============================================================================

class TestBaseMetaTitleValidation:
    """Test title field validation (non-empty string)."""

    def test_title_single_character(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that single-character title is valid."""
        valid_base_meta_dict["title"] = "A"
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.title == "A"

    def test_title_long_string(self, valid_base_meta_dict: dict) -> None:
        """Test that long title is valid."""
        long_title = "A" * 1000
        valid_base_meta_dict["title"] = long_title
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.title == long_title

    def test_title_empty_string_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that empty title raises ValidationError."""
        valid_base_meta_dict["title"] = ""
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "at least 1" in str(exc_info.value)

    def test_title_with_special_characters(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test title with special characters."""
        valid_base_meta_dict["title"] = "NGC 1234 [Test-Obj_2.5] α=0.5"
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.title == "NGC 1234 [Test-Obj_2.5] α=0.5"


# ============================================================================
# Pydantic Validation: Redshift Field
# ============================================================================

class TestBaseMetaRedshiftValidation:
    """Test redshift (z) field validation (z >= 0)."""

    def test_z_zero(self, valid_base_meta_dict: dict) -> None:
        """Test that z=0 is valid."""
        valid_base_meta_dict["z"] = 0.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.z == 0.0

    def test_z_large_value(self, valid_base_meta_dict: dict) -> None:
        """Test that large redshift is valid."""
        valid_base_meta_dict["z"] = 10.5
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.z == 10.5

    def test_z_negative_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that z < 0 raises ValidationError."""
        valid_base_meta_dict["z"] = -0.1
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "greater than or equal to 0" in str(exc_info.value)


# ============================================================================
# Pydantic Validation: Luminosity Distance Field
# ============================================================================

class TestBaseMetaLuminosityDistanceValidation:
    """Test luminosity_distance_mpc field validation (dist >= 0)."""

    def test_luminosity_distance_zero(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that distance=0 is valid."""
        valid_base_meta_dict["luminosity_distance_mpc"] = 0.0
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.luminosity_distance_mpc == 0.0

    def test_luminosity_distance_large_value(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that large distance is valid."""
        valid_base_meta_dict["luminosity_distance_mpc"] = 1e6
        meta = BaseMeta(**valid_base_meta_dict)
        assert meta.luminosity_distance_mpc == 1e6

    def test_luminosity_distance_negative_raises_error(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that distance < 0 raises ValidationError."""
        valid_base_meta_dict["luminosity_distance_mpc"] = -1.0
        with pytest.raises(ValidationError) as exc_info:
            BaseMeta(**valid_base_meta_dict)
        assert "greater than or equal to 0" in str(exc_info.value)


# ============================================================================
# JSON Serialization Tests
# ============================================================================

class TestBaseMetaJSONSerialization:
    """Test JSON serialization of BaseMeta."""

    def test_model_dump_json(self, valid_base_meta_dict: dict) -> None:
        """Test that model_dump_json produces valid JSON."""
        meta = BaseMeta(**valid_base_meta_dict)
        json_str = meta.model_dump_json()
        assert isinstance(json_str, str)
        # Verify it's valid JSON by parsing
        parsed = json.loads(json_str)
        assert parsed["title"] == "Test Spectrum"

    def test_model_dump(self, valid_base_meta_dict: dict) -> None:
        """Test that model_dump produces dict with string path."""
        meta = BaseMeta(**valid_base_meta_dict)
        dumped = meta.model_dump()
        assert isinstance(dumped, dict)
        assert "path" in dumped
        assert "title" in dumped
        assert dumped["title"] == "Test Spectrum"

    def test_json_roundtrip(self, valid_base_meta_dict: dict) -> None:
        """Test JSON serialization and deserialization roundtrip."""
        original = BaseMeta(**valid_base_meta_dict)
        json_str = original.model_dump_json()
        # Parse JSON and reconstruct from dict to handle path serialization
        data = json.loads(json_str)
        restored = BaseMeta(**data)
        assert restored.title == original.title
        assert restored.ra == original.ra
        assert restored.dec == original.dec
        assert restored.z == original.z
        assert restored.luminosity_distance_mpc == (
            original.luminosity_distance_mpc
        )

    def test_json_to_file_and_load(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test saving to JSON file and loading back."""
        meta = BaseMeta(**valid_base_meta_dict)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json_file = Path(f.name)
            f.write(meta.model_dump_json())

        try:
            with open(json_file) as f:
                json_str = f.read()
            # Parse JSON and reconstruct from dict to handle path
            data = json.loads(json_str)
            restored = BaseMeta(**data)
            assert restored.title == meta.title
            assert restored.ra == meta.ra
            assert restored.dec == meta.dec
        finally:
            json_file.unlink()

    def test_json_contains_all_fields(
        self, valid_base_meta_dict: dict
    ) -> None:
        """Test that JSON output contains all expected fields."""
        meta = BaseMeta(**valid_base_meta_dict)
        json_str = meta.model_dump_json()
        parsed = json.loads(json_str)
        required_fields = [
            "path", "title", "ra", "dec", "z", "luminosity_distance_mpc"
        ]
        for field in required_fields:
            assert field in parsed, f"Field '{field}' missing from JSON"


# ============================================================================
# SDSSMeta Tests
# ============================================================================

class TestSDSSMetaInstantiation:
    """Test SDSSMeta specific behavior."""

    def test_create_sdss_meta(self, valid_sdss_meta_dict: dict) -> None:
        """Test creating SDSSMeta with SDSS-specific fields."""
        meta = SDSSMeta(**valid_sdss_meta_dict)
        assert meta.plate == 1234
        assert meta.fiber == 567
        assert meta.mjd == 55000
        # PAQS fields should be None
        assert meta.obj_nme is None
        assert meta.obj_uid is None
        assert meta.specuid is None

    def test_sdss_meta_inherits_base_fields(
        self, valid_sdss_meta_dict: dict
    ) -> None:
        """Test that SDSSMeta inherits base fields correctly."""
        meta = SDSSMeta(**valid_sdss_meta_dict)
        assert meta.title == "Test Spectrum"
        assert meta.ra == 180.0
        assert meta.dec == 0.0

    def test_sdss_meta_without_optional_fields(
        self, fits_file: Path
    ) -> None:
        """Test SDSSMeta with only required fields."""
        meta = SDSSMeta(
            path=str(fits_file),
            title="Test",
            ra=180.0,
            dec=0.0,
            z=0.5,
            luminosity_distance_mpc=2000.0,
        )
        assert meta.plate is None
        assert meta.fiber is None
        assert meta.mjd is None


# ============================================================================
# PAQSMeta Tests
# ============================================================================

class TestPAQSMetaInstantiation:
    """Test PAQSMeta specific behavior."""

    def test_create_paqs_meta(self, valid_paqs_meta_dict: dict) -> None:
        """Test creating PAQSMeta with PAQS-specific fields."""
        meta = PAQSMeta(**valid_paqs_meta_dict)
        assert meta.obj_nme == "NGC_1234"
        assert meta.obj_uid == 999
        assert meta.specuid == 777
        # SDSS fields should be None
        assert meta.plate is None
        assert meta.fiber is None
        assert meta.mjd is None

    def test_paqs_meta_inherits_base_fields(
        self, valid_paqs_meta_dict: dict
    ) -> None:
        """Test that PAQSMeta inherits base fields correctly."""
        meta = PAQSMeta(**valid_paqs_meta_dict)
        assert meta.title == "Test Spectrum"
        assert meta.ra == 180.0
        assert meta.dec == 0.0

    def test_paqs_meta_obj_nme_validation(
        self, valid_paqs_meta_dict: dict
    ) -> None:
        """Test that obj_nme cannot be empty string."""
        valid_paqs_meta_dict["obj_nme"] = ""
        with pytest.raises(ValidationError) as exc_info:
            PAQSMeta(**valid_paqs_meta_dict)
        assert "at least 1" in str(exc_info.value)

    def test_paqs_meta_without_optional_fields(
        self, fits_file: Path
    ) -> None:
        """Test PAQSMeta with only required fields."""
        meta = PAQSMeta(
            path=str(fits_file),
            title="Test",
            ra=180.0,
            dec=0.0,
            z=0.5,
            luminosity_distance_mpc=2000.0,
        )
        assert meta.obj_nme is None
        assert meta.obj_uid is None
        assert meta.specuid is None


# ============================================================================
# Cross-Metadata Type Tests
# ============================================================================

class TestMetadataTypeDifferences:
    """Test differences between SDSSMeta and PAQSMeta."""

    def test_sdss_and_paqs_not_interchangeable(
        self, valid_sdss_meta_dict: dict, valid_paqs_meta_dict: dict
    ) -> None:
        """Test that SDSS and PAQS metadata are different types."""
        sdss_meta = SDSSMeta(**valid_sdss_meta_dict)
        paqs_meta = PAQSMeta(**valid_paqs_meta_dict)
        assert type(sdss_meta) != type(paqs_meta)
        assert isinstance(sdss_meta, BaseMeta)
        assert isinstance(paqs_meta, BaseMeta)

    def test_sdss_meta_class_vars_none(
        self, valid_sdss_meta_dict: dict
    ) -> None:
        """Test that SDSSMeta class variables are None for PAQS fields."""
        assert SDSSMeta.obj_nme is None
        assert SDSSMeta.obj_uid is None
        assert SDSSMeta.specuid is None
