import pytest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle.utils.arg_validator import is_kaggle_model, is_flux_gguf_model, validate_args

class TestValidateArgs:
    # ...existing code...

    @pytest.mark.parametrize(
        "model_id,img_width,img_height,expected_error,expected_msg",
        [
            ("flux-gguf-q4", 500, 250, True, "FLUX models require image dimensions divisible by 16"),
            ("stabilityai/stable-diffusion-xl", 500, 250, True, "Stable Diffusion models require image dimensions divisible by 8"),
            ("flux-gguf-q4", 512, 512, False, None),
            ("stabilityai/stable-diffusion-xl", 512, 512, False, None),
        ]
    )
    def test_image_dimension_validation(self, model_id, img_width, img_height, expected_error, expected_msg):
        args = MagicMock()
        args.model_id = model_id
        args.prompts = ["test"]
        args.prompts_file = None
        args.img_width = img_width
        args.img_height = img_height
        args.steps = 10
        args.guidance = 7.5
        args.precision = "fp16"
        args.refiner_model_id = None
        args.refiner_guidance = None
        args.refiner_steps = None
        args.refiner_precision = None
        args.refiner_negative_prompt = None
        args.gpu = True
        args.notebook = None
        args.kernel_path = None
        args.dest = None
        args.output_base_dir = None
        args.negative_prompt = None
        args.hf_token = None
        args.refiner_model = None
        if expected_error:
            with pytest.raises(ValueError, match=expected_msg):
                validate_args(args)
        else:
            validate_args(args)



import pytest
from unittest.mock import patch, MagicMock
from imggenhub.kaggle.utils.arg_validator import is_kaggle_model, is_flux_gguf_model, validate_args


class TestIsKaggleModel:
    """Comprehensive tests for is_kaggle_model function."""

    def test_kaggle_models_return_true(self):
        """Test that known non-HuggingFace org models return True."""
        kaggle_models = [
            "leventecsibi/stable-diffusion-model",
            "someuser/custom-model",
            "randomuser123/my-model",
            "kaggleuser/model-name",
            "independent/model-v1"
        ]
        for model_id in kaggle_models:
            assert is_kaggle_model(model_id) is True, f"Failed for {model_id}"

    def test_huggingface_orgs_return_false(self):
        """Test that known HuggingFace org models return False."""
        hf_orgs = [
            "stabilityai/stable-diffusion-xl-base-1.0",
            "black-forest-labs/FLUX.1-schnell",
            "runwayml/stable-diffusion-v1-5",
            "compvis/stable-diffusion",
            "openai/clip-vit-large-patch14",
            "google/siglip-so400m-patch14-384",
            "microsoft/DialoGPT-medium",
            "facebook/opt-1.3b",
            "huggingface/zephyr-7b-beta",
            "meta-llama/Llama-2-7b-hf",
            "anthropic/unknown-model",
            "eleutherai/gpt-neo-1.3B",
            "bigscience/bloom-560m",
            "bigcode/starcoder",
            "salesforce/codegen-350M-mono",
            "amazon/Qwen2-7B",
            "nvidia/Llama-3.1-8B-Instruct",
            "intel/neural-chat-7b-v3-1",
            "apple/OpenELM-270M",
            "tencent/unknown-model",
            "baidu/ernie-3.0-base-zh",
            "alibaba/Qwen-7B",
            "bytedance/unknown-model"
        ]
        for model_id in hf_orgs:
            assert is_kaggle_model(model_id) is False, f"Failed for {model_id}"

    def test_no_slash_returns_false(self):
        """Test that model IDs without '/' return False."""
        invalid_models = [
            "stabilityai",
            "model-name-only",
            "",
            "singleword",
            "user/",  # invalid: empty model name
            "/model",  # invalid: empty owner
        ]
        for model_id in invalid_models:
            assert is_kaggle_model(model_id) is False, f"Failed for {model_id}"

    def test_case_insensitive_hf_orgs(self):
        """Test that HuggingFace org detection is case insensitive."""
        case_variations = [
            "StabilityAI/model",
            "BLACK-FOREST-LABS/model",
            "RunwayML/model",
            "STABILITYAI/model",
            "stabilityai/model"
        ]
        for model_id in case_variations:
            assert is_kaggle_model(model_id) is False, f"Failed for {model_id}"

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        edge_cases = [
            ("user/", False),  # trailing slash - invalid
            ("/model", False),  # leading slash - invalid
            ("user/model/extra", True),  # extra path components
            ("user/model.git", True),  # .git suffix
            ("user/model@v1.0", True),  # version suffix
            ("user/model#branch", True),  # branch suffix
            ("no/slash/here", True),  # multiple slashes, valid format
        ]
        for model_id, expected in edge_cases:
            assert is_kaggle_model(model_id) == expected, f"Failed for {model_id}"


class TestIsFluxGguFModel:
    """Comprehensive tests for is_flux_gguf_model function."""

    def test_flux_gguf_models_return_true(self):
        """Test that FLUX GGUF models return True."""
        flux_gguf_models = [
            "city96/FLUX.1-schnell-gguf",
            "flux-gguf-model",
            "someuser/flux-1-dev-gguf",
            "FLUX.1-dev-Q4_0.gguf",
            "flux-schnell-q8_0",
            "FLUX-Q4.gguf",
            "flux-1-schnell-gguf-q4",
            "custom-flux-gguf-model"
        ]
        for model_id in flux_gguf_models:
            assert is_flux_gguf_model(model_id) is True, f"Failed for {model_id}"

    def test_non_flux_models_return_false(self):
        """Test that non-FLUX models return False."""
        non_flux_models = [
            "stabilityai/stable-diffusion-xl",
            "black-forest-labs/FLUX.1-schnell",
            "runwayml/stable-diffusion-v1-5",
            "compvis/stable-diffusion",
            "regular-model-name",
            "some-model",
            "flux",  # just "flux" without gguf/q4/q8
            "flux-model",  # just "flux" without gguf/q4/q8
        ]
        for model_id in non_flux_models:
            assert is_flux_gguf_model(model_id) is False, f"Failed for {model_id}"

    def test_empty_and_none_inputs(self):
        """Test empty and None inputs."""
        assert is_flux_gguf_model("") is False
        assert is_flux_gguf_model(None) is False

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        case_variations = [
            "CITY96/FLUX.1-SCHNELL-GGUF",
            "Flux-Gguf-Model",
            "FLUX-Q4_0",
            "flux-1-dev-gguf",
            "FLUX.1-SCHNELL-Q8_0"
        ]
        for model_id in case_variations:
            assert is_flux_gguf_model(model_id) is True, f"Failed for {model_id}"

    def test_partial_matches(self):
        """Test partial keyword matches."""
        partial_matches = [
            "my-flux-gguf-custom",
            "flux-gguf-v2",
            "flux-q4-model",
            "flux-q8-model",
            "flux-1-gguf",
            "flux-schnell-gguf"
        ]
        for model_id in partial_matches:
            assert is_flux_gguf_model(model_id) is True, f"Failed for {model_id}"


class TestValidateArgs:
    """Comprehensive tests for validate_args function."""

    def create_mock_args(self, **kwargs):
        """Helper to create mock args object."""
        args = MagicMock()
        # Set default valid values
        args.model_id = "leventecsibi/test-model"
        args.prompt = None
        args.prompts = ["test prompt"]
        args.prompts_file = None
        args.img_width = 512
        args.img_height = 512
        args.steps = 20
        args.guidance = 7.5
        args.precision = "fp16"
        args.model_filename = None
        args.diffusion_repo_id = None
        args.diffusion_filename = None
        args.refiner_model_id = None
        args.refiner_precision = None
        args.refiner_guidance = None
        args.refiner_steps = None

        # Override with provided kwargs
        for key, value in kwargs.items():
            setattr(args, key, value)
        return args

    def test_valid_args_pass(self):
        """Test that valid arguments pass validation."""
        args = self.create_mock_args()
        # Should not raise any exception
        validate_args(args)

    def test_missing_model_id_without_diffusion_params_raises_error(self):
        """Test that missing model_id without diffusion params raises error."""
        args = self.create_mock_args(model_id=None, diffusion_repo_id=None, diffusion_filename=None)
        with pytest.raises(ValueError, match="--model_id is required"):
            validate_args(args)

    def test_missing_model_id_with_diffusion_repo_raises_error(self):
        """Test that missing model_id with diffusion_repo_id raises error."""
        args = self.create_mock_args(model_id=None, diffusion_repo_id="city96/FLUX.1-schnell-gguf")
        with pytest.raises(ValueError, match="--model_id is required"):
            validate_args(args)

    def test_missing_model_id_with_diffusion_filename_raises_error(self):
        """Test that missing model_id with diffusion_filename raises error."""
        args = self.create_mock_args(model_id=None, diffusion_filename="flux1-schnell-Q4_0.gguf")
        with pytest.raises(ValueError, match="--model_id is required"):
            validate_args(args)

    def test_model_filename_without_model_id_raises_error(self):
        """Test that model_filename without model_id raises error."""
        args = self.create_mock_args(model_id=None, model_filename="model.gguf")
        with pytest.raises(ValueError, match="--model_id is required when --model_filename is provided"):
            validate_args(args)

    def test_no_prompts_raises_error(self):
        """Test that no prompts raises error."""
        args = self.create_mock_args(prompt=None, prompts=None, prompts_file=None)
        with pytest.raises(ValueError, match="No prompts provided"):
            validate_args(args)

    def test_missing_img_width_raises_error(self):
        """Test that missing img_width raises error."""
        args = self.create_mock_args(img_width=None)
        with pytest.raises(ValueError, match="Both --img_width and --img_height are required"):
            validate_args(args)

    def test_missing_img_height_raises_error(self):
        """Test that missing img_height raises error."""
        args = self.create_mock_args(img_height=None)
        with pytest.raises(ValueError, match="Both --img_width and --img_height are required"):
            validate_args(args)

    def test_flux_gguf_invalid_dimensions_raises_error(self):
        """Test that FLUX GGUF models with invalid dimensions raise error."""
        args = self.create_mock_args(
            model_id="city96/FLUX.1-schnell-gguf",
            img_width=511,  # not divisible by 16
            img_height=512
        )
        with pytest.raises(ValueError, match="FLUX models require image dimensions divisible by 16"):
            validate_args(args)

    def test_flux_gguf_valid_dimensions_pass(self):
        """Test that FLUX GGUF models with valid dimensions pass."""
        args = self.create_mock_args(
            model_id="city96/FLUX.1-schnell-gguf",
            img_width=512,  # divisible by 16
            img_height=512
        )
        validate_args(args)  # Should not raise

    def test_sd_invalid_dimensions_raises_error(self):
        """Test that SD models with invalid dimensions raise error."""
        args = self.create_mock_args(
            model_id="stabilityai/stable-diffusion-xl",
            img_width=511,  # not divisible by 8
            img_height=512
        )
        with pytest.raises(ValueError, match="Stable Diffusion models require image dimensions divisible by 8"):
            validate_args(args)

    def test_sd_valid_dimensions_pass(self):
        """Test that SD models with valid dimensions pass."""
        args = self.create_mock_args(
            model_id="stabilityai/stable-diffusion-xl",
            img_width=512,  # divisible by 8
            img_height=512
        )
        validate_args(args)  # Should not raise

    def test_missing_steps_raises_error(self):
        """Test that missing steps raises error."""
        args = self.create_mock_args(steps=None)
        with pytest.raises(ValueError, match="--steps is required"):
            validate_args(args)

    def test_missing_guidance_raises_error(self):
        """Test that missing guidance raises error."""
        args = self.create_mock_args(guidance=None)
        with pytest.raises(ValueError, match="--guidance is required"):
            validate_args(args)

    def test_missing_precision_raises_error(self):
        """Test that missing precision raises error."""
        args = self.create_mock_args(precision=None)
        with pytest.raises(ValueError, match="--precision is required"):
            validate_args(args)

    @patch('imggenhub.kaggle.utils.precision_validator.PrecisionValidator')
    @patch.dict('os.environ', {'HF_TOKEN': 'test-token'})
    def test_invalid_precision_for_non_kaggle_model_raises_error(self, mock_precision_validator_class):
        """Test that invalid precision for non-Kaggle model raises error."""
        mock_detector = MagicMock()
        mock_detector.detect_available_variants.return_value = ['fp16', 'fp32']
        mock_precision_validator_class.return_value = mock_detector

        args = self.create_mock_args(
            model_id="stabilityai/stable-diffusion-xl",  # HF org model
            precision="int8"  # not available
        )
        with pytest.raises(ValueError, match="Precision 'int8' not available"):
            validate_args(args)

    @patch('imggenhub.kaggle.utils.precision_validator.PrecisionValidator')
    @patch.dict('os.environ', {'HF_TOKEN': 'test-token'})
    def test_valid_precision_for_non_kaggle_model_passes(self, mock_precision_validator_class):
        """Test that valid precision for non-Kaggle model passes."""
        mock_detector = MagicMock()
        mock_detector.detect_available_variants.return_value = ['fp16', 'fp32']
        mock_precision_validator_class.return_value = mock_detector

        args = self.create_mock_args(
            model_id="stabilityai/stable-diffusion-xl",  # HF org model
            precision="fp16"  # available
        )
        validate_args(args)  # Should not raise

    def test_kaggle_model_skips_precision_validation(self):
        """Test that Kaggle models skip precision validation."""
        args = self.create_mock_args(
            model_id="leventecsibi/custom-model",  # Kaggle model (not HF org)
            precision="invalid-precision"
        )
        validate_args(args)  # Should not raise

    def test_flux_model_skips_precision_validation(self):
        """Test that FLUX models skip precision validation."""
        args = self.create_mock_args(
            model_id="black-forest-labs/FLUX.1-schnell",  # FLUX model
            precision="invalid-precision"
        )
        validate_args(args)  # Should not raise

    def test_auto_precision_skips_validation(self):
        """Test that 'auto' precision skips validation."""
        args = self.create_mock_args(
            model_id="someuser/custom-model",
            precision="auto"
        )
        validate_args(args)  # Should not raise

    def test_refiner_without_model_id_raises_error(self):
        """Test that refiner_precision without refiner_model_id raises error."""
        args = self.create_mock_args(
            refiner_model_id=None,
            refiner_precision="fp16"
        )
        with pytest.raises(ValueError, match="--refiner_model_id is required when specifying --refiner_precision"):
            validate_args(args)

    @patch('imggenhub.kaggle.utils.precision_validator.PrecisionValidator')
    @patch.dict('os.environ', {'HF_TOKEN': 'test-token'})
    def test_invalid_refiner_precision_raises_error(self, mock_precision_validator_class):
        """Test that invalid refiner precision raises error, but guidance is checked first."""
        mock_detector = MagicMock()
        mock_detector.detect_available_variants.return_value = ['fp16', 'fp32']
        mock_precision_validator_class.return_value = mock_detector

        args = self.create_mock_args(
            refiner_model_id="someuser/refiner-model",
            refiner_precision="int8"  # not available
        )
        # The function checks for refiner_guidance first, so expect that error
        with pytest.raises(ValueError, match="--refiner_guidance is required when using a refiner model"):
            validate_args(args)

    def test_refiner_without_guidance_raises_error(self):
        """Test that refiner without guidance raises error."""
        args = self.create_mock_args(
            refiner_model_id="someuser/refiner-model",
            refiner_guidance=None
        )
        with pytest.raises(ValueError, match="--refiner_guidance is required when using a refiner model"):
            validate_args(args)

    def test_refiner_without_steps_raises_error(self):
        """Test that refiner without steps raises error, but guidance is checked first."""
        args = self.create_mock_args(
            refiner_model_id="someuser/refiner-model",
            refiner_steps=None
        )
        # The function checks for refiner_guidance first, so expect that error
        with pytest.raises(ValueError, match="--refiner_guidance is required when using a refiner model"):
            validate_args(args)

    @patch('builtins.print')  # Mock print to capture warnings
    def test_model_filename_warning_for_sd_model(self, mock_print):
        """Test that model_filename warning is printed for SD models."""
        args = self.create_mock_args(
            model_id="stabilityai/stable-diffusion-xl",
            model_filename="unnecessary.gguf"
        )
        validate_args(args)
        # Check that warning was printed
        mock_print.assert_called()

    @patch('builtins.print')  # Mock print to capture warnings
    def test_model_filename_warning_for_flux_bf16(self, mock_print):
        """Test that model_filename warning is printed for FLUX BF16 models."""
        args = self.create_mock_args(
            model_id="black-forest-labs/FLUX.1-schnell",
            model_filename="unnecessary.gguf"
        )
        validate_args(args)
        # Check that warning was printed
        mock_print.assert_called()

    def test_model_filename_allowed_for_flux_gguf(self):
        """Test that model_filename is allowed for FLUX GGUF models."""
        args = self.create_mock_args(
            model_id="city96/FLUX.1-schnell-gguf",
            model_filename="flux1-schnell-Q4_0.gguf"
        )
        validate_args(args)  # Should not raise or print warning