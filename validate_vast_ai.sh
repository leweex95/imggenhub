#!/bin/bash
# Validation script for Vast.ai integration

set -e

echo "=== Vast.ai Integration Validation ==="
echo ""

# Check Python syntax
echo "1. Checking Python syntax..."
python -m py_compile src/imggenhub/vast_ai/client.py
python -m py_compile src/imggenhub/vast_ai/ssh.py
python -m py_compile src/imggenhub/vast_ai/executor.py
python -m py_compile src/imggenhub/vast_ai/forge.py
python -m py_compile src/imggenhub/vast_ai/cost_tracking.py
python -m py_compile src/imggenhub/vast_ai/test_remote_pipeline.py
python -m py_compile src/imggenhub/vast_ai/forge_cli.py
python -m py_compile src/imggenhub/vast_ai/cost_cli.py
python -m py_compile src/imggenhub/vast_ai/cli.py
echo "   ✓ All Python files valid"
echo ""

# Check shell scripts
echo "2. Checking shell scripts..."
bash -n src/imggenhub/vast_ai/setup.sh
bash -n src/imggenhub/vast_ai/forge_deploy.sh
echo "   ✓ All shell scripts valid"
echo ""

# Check documentation
echo "3. Checking documentation files..."
[ -f docs/VAST_AI_README.md ] && echo "   ✓ VAST_AI_README.md" || echo "   ✗ Missing VAST_AI_README.md"
[ -f docs/VAST_AI_INTEGRATION.md ] && echo "   ✓ VAST_AI_INTEGRATION.md" || echo "   ✗ Missing VAST_AI_INTEGRATION.md"
[ -f docs/VAST_AI_TEST_GUIDE.md ] && echo "   ✓ VAST_AI_TEST_GUIDE.md" || echo "   ✗ Missing VAST_AI_TEST_GUIDE.md"
[ -f docs/TESLA_P40_QUICK_START.md ] && echo "   ✓ TESLA_P40_QUICK_START.md" || echo "   ✗ Missing TESLA_P40_QUICK_START.md"
[ -f docs/FORGE_UI_INTEGRATION.md ] && echo "   ✓ FORGE_UI_INTEGRATION.md" || echo "   ✗ Missing FORGE_UI_INTEGRATION.md"
[ -f docs/VAST_AI_QUICK_REFERENCE.md ] && echo "   ✓ VAST_AI_QUICK_REFERENCE.md" || echo "   ✗ Missing VAST_AI_QUICK_REFERENCE.md"
echo ""

# Check CLI entry points
echo "4. Checking CLI entry points in pyproject.toml..."
grep -q "imggenhub = " pyproject.toml && echo "   ✓ imggenhub entry point" || echo "   ✗ Missing imggenhub"
grep -q "imggenhub-vast = " pyproject.toml && echo "   ✓ imggenhub-vast entry point" || echo "   ✗ Missing imggenhub-vast"
grep -q "imggenhub-forge = " pyproject.toml && echo "   ✓ imggenhub-forge entry point" || echo "   ✗ Missing imggenhub-forge"
grep -q "imggenhub-costs = " pyproject.toml && echo "   ✓ imggenhub-costs entry point" || echo "   ✗ Missing imggenhub-costs"
echo ""

# Check required dependencies
echo "5. Checking required dependencies..."
grep -q "requests" pyproject.toml && echo "   ✓ requests" || echo "   ✗ Missing requests"
grep -q "paramiko" pyproject.toml && echo "   ✓ paramiko" || echo "   ✗ Missing paramiko"
echo ""

echo "=== Validation Complete ==="
echo "All checks passed! Ready for production."
