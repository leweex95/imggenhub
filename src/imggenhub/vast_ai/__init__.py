"""Vast.ai integration for remote GPU instance management."""

from imggenhub.vast_ai.api import VastAiClient, VastInstance
from imggenhub.vast_ai.auto_deploy import AutoDeploymentOrchestrator, SearchCriteria
from imggenhub.vast_ai.types import RemoteRunResult
from imggenhub.vast_ai.config import VastAiConfig, get_vast_ai_config
from imggenhub.vast_ai.deployments import VastApiError, VastDeploymentClient
from imggenhub.vast_ai.dummy import DummyDeploymentResult, DummyDeploymentRunner
from imggenhub.vast_ai.pipelines import PipelineExecution, PipelineOrchestrator
from imggenhub.vast_ai.orchestration import RemoteExecutor
from imggenhub.vast_ai.ssh import SSHClient
from imggenhub.vast_ai.utils import CostEstimator, InstanceManager, RunLogger

__all__ = [
    "VastAiClient",
    "VastInstance",
    "SSHClient",
    "RemoteExecutor",
    "get_vast_ai_config",
    "VastAiConfig",
    "CostEstimator",
    "RunLogger",
    "InstanceManager",
    "VastDeploymentClient",
    "VastApiError",
    "DummyDeploymentRunner",
    "DummyDeploymentResult",
    "RemoteRunResult",
    "PipelineOrchestrator",
    "PipelineExecution",
    "AutoDeploymentOrchestrator",
    "SearchCriteria",
]
