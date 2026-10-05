#!/usr/bin/env python3
"""CDK entry point. All settings come from context (``-c key=value``); see README.

cd infra
uv run --group infra npx aws-cdk@2 synth -c artifact_version=2026-10-05.1
"""

import os

import aws_cdk as cdk

from wc2026_infra.stack import ServiceConfig, Wc2026ServiceStack

app = cdk.App()


def ctx(key: str, default: str | None = None) -> str | None:
    value = app.node.try_get_context(key)
    return default if value is None or value == "" else str(value)


env_name = ctx("env_name", "demo") or "demo"
config = ServiceConfig(
    env_name=env_name,
    github_repo=ctx("github_repo", "aicgoraya/wc2026") or "",
    github_environment=ctx("github_environment", "production") or "",
    artifact_version=ctx("artifact_version", "unset") or "unset",
    artifact_prefix=ctx("artifact_prefix", "artifacts") or "artifacts",
    image=ctx("image"),
    desired_count=int(ctx("desired_count", "0") or 0),
    cpu=int(ctx("cpu", "256") or 256),
    memory_mib=int(ctx("memory_mib", "512") or 512),
    log_retention_days=int(ctx("log_retention_days", "30") or 30),
    certificate_arn=ctx("certificate_arn"),
    create_oidc_provider=(ctx("create_oidc_provider", "true") or "true").lower() == "true",
)

Wc2026ServiceStack(
    app,
    f"wc2026-{env_name}",
    config=config,
    # Region/account come from the CLI's credentials at deploy time; the stack
    # itself is environment-agnostic so `cdk synth` needs no AWS access.
    env=cdk.Environment(
        account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
        region=ctx("region") or os.environ.get("CDK_DEFAULT_REGION"),
    )
    if ctx("region")
    else None,
)
cdk.Tags.of(app).add("project", "wc2026")
cdk.Tags.of(app).add("environment", env_name)
app.synth()
