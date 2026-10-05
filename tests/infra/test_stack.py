"""Infrastructure checks on the synthesized CloudFormation template (no AWS access)."""

import shutil
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("aws_cdk")
if shutil.which("node") is None:  # aws-cdk-lib drives a Node.js runtime (jsii)
    pytest.skip("node is required to synthesize the CDK stack", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "infra"))

import aws_cdk as cdk
from aws_cdk.assertions import Match, Template
from wc2026_infra.stack import ServiceConfig, Wc2026ServiceStack


def synth(**overrides: Any) -> Template:
    config = ServiceConfig(**{"artifact_version": "v-test", **overrides})
    return Template.from_stack(Wc2026ServiceStack(cdk.App(), "wc2026-test", config=config))


@pytest.fixture(scope="module")
def template() -> Template:
    return synth()


def test_resource_inventory_is_exactly_what_the_readme_lists(template: Template) -> None:
    expected = {
        "AWS::S3::Bucket": 1,
        "AWS::ECR::Repository": 1,
        "AWS::Logs::LogGroup": 1,
        "AWS::ECS::Cluster": 1,
        "AWS::ECS::TaskDefinition": 1,
        "AWS::ECS::Service": 1,
        "AWS::ElasticLoadBalancingV2::LoadBalancer": 1,
        "AWS::ElasticLoadBalancingV2::TargetGroup": 1,
        "AWS::ElasticLoadBalancingV2::Listener": 1,
        "AWS::EC2::VPC": 1,
        "AWS::EC2::Subnet": 2,
        "AWS::EC2::SecurityGroup": 2,
        "AWS::IAM::Role": 3,
        "AWS::IAM::OIDCProvider": 1,
        "AWS::EC2::NatGateway": 0,
        "AWS::Lambda::Function": 0,
        "AWS::EC2::EIP": 0,
    }
    for resource_type, count in expected.items():
        template.resource_count_is(resource_type, count)


def test_artifact_bucket_is_private_encrypted_and_retained(template: Template) -> None:
    template.has_resource(
        "AWS::S3::Bucket",
        {
            "DeletionPolicy": "Retain",
            "Properties": {
                "PublicAccessBlockConfiguration": {
                    "BlockPublicAcls": True,
                    "BlockPublicPolicy": True,
                    "IgnorePublicAcls": True,
                    "RestrictPublicBuckets": True,
                },
                "BucketEncryption": Match.any_value(),
                "VersioningConfiguration": {"Status": "Enabled"},
            },
        },
    )


def test_images_are_immutable_and_scanned(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ECR::Repository",
        {"ImageTagMutability": "IMMUTABLE", "ImageScanningConfiguration": {"ScanOnPush": True}},
    )


def test_logs_have_bounded_retention(template: Template) -> None:
    template.has_resource_properties("AWS::Logs::LogGroup", {"RetentionInDays": 30})
    with pytest.raises(Exception, match="log_retention_days"):
        synth(log_retention_days=0)


def test_task_is_small_x86_fargate_with_pinned_artifact_and_cloudwatch_logs(
    template: Template,
) -> None:
    template.has_resource_properties(
        "AWS::ECS::TaskDefinition",
        {
            "Cpu": "256",
            "Memory": "512",
            "RequiresCompatibilities": ["FARGATE"],
            "RuntimePlatform": {"CpuArchitecture": "X86_64", "OperatingSystemFamily": "LINUX"},
            "ContainerDefinitions": [
                Match.object_like(
                    {
                        "Name": "api",
                        "PortMappings": [Match.object_like({"ContainerPort": 8080})],
                        "LogConfiguration": Match.object_like({"LogDriver": "awslogs"}),
                        "Environment": Match.array_with(
                            [
                                {"Name": "WC2026_ARTIFACT_SOURCE", "Value": "s3"},
                                {"Name": "WC2026_ARTIFACT_VERSION", "Value": "v-test"},
                            ]
                        ),
                    }
                )
            ],
        },
    )


def test_container_gets_no_secrets_or_static_credentials(template: Template) -> None:
    definitions = template.find_resources("AWS::ECS::TaskDefinition")
    (task,) = definitions.values()
    (container,) = task["Properties"]["ContainerDefinitions"]
    assert "Secrets" not in container
    names = {entry["Name"] for entry in container["Environment"]}
    assert not {n for n in names if "KEY" in n or "SECRET" in n or "TOKEN" in n}


def test_service_starts_at_zero_with_rollback_and_only_alb_ingress(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ECS::Service",
        {
            "DesiredCount": 0,
            "LaunchType": "FARGATE",
            "DeploymentConfiguration": Match.object_like(
                {"DeploymentCircuitBreaker": {"Enable": True, "Rollback": True}}
            ),
        },
    )
    # the only ingress into the task security group is from the ALB's group on 8080
    template.resource_count_is("AWS::EC2::SecurityGroupIngress", 1)
    template.has_resource_properties(
        "AWS::EC2::SecurityGroupIngress",
        {
            "FromPort": 8080,
            "ToPort": 8080,
            "IpProtocol": "tcp",
            "SourceSecurityGroupId": Match.any_value(),
        },
    )
    groups = template.find_resources("AWS::EC2::SecurityGroup")
    open_to_world = [
        g
        for g in groups.values()
        if any(
            r.get("CidrIp") == "0.0.0.0/0" for r in g["Properties"].get("SecurityGroupIngress", [])
        )
    ]
    assert len(open_to_world) == 1  # the load balancer, and nothing else
    assert [r["FromPort"] for r in open_to_world[0]["Properties"]["SecurityGroupIngress"]] == [80]


def test_load_balancer_checks_readiness_not_liveness(template: Template) -> None:
    template.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::TargetGroup",
        {"HealthCheckPath": "/ready", "Matcher": {"HttpCode": "200"}, "TargetType": "ip"},
    )


def test_http_only_without_a_certificate_and_https_with_one() -> None:
    synth().has_resource_properties(
        "AWS::ElasticLoadBalancingV2::Listener", {"Port": 80, "Protocol": "HTTP"}
    )
    with_cert = synth(certificate_arn="arn:aws:acm:us-east-1:111111111111:certificate/abc")
    with_cert.resource_count_is("AWS::ElasticLoadBalancingV2::Listener", 2)
    with_cert.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::Listener", {"Port": 443, "Protocol": "HTTPS"}
    )
    with_cert.has_resource_properties(
        "AWS::ElasticLoadBalancingV2::Listener",
        {"Port": 80, "DefaultActions": [Match.object_like({"Type": "redirect"})]},
    )


def policy_statements(template: Template, role_fragment: str) -> list[dict[str, Any]]:
    statements: list[dict[str, Any]] = []
    for policy in template.find_resources("AWS::IAM::Policy").values():
        if any(role_fragment in str(ref) for ref in policy["Properties"]["Roles"]):
            statements.extend(policy["Properties"]["PolicyDocument"]["Statement"])
    return statements


def actions_of(statements: list[dict[str, Any]]) -> set[str]:
    found: set[str] = set()
    for statement in statements:
        action = statement["Action"]
        found.update([action] if isinstance(action, str) else action)
    return found


def test_application_role_can_only_read_the_artifact_prefix(template: Template) -> None:
    statements = policy_statements(template, "TaskRole")
    assert actions_of(statements) == {"s3:GetObject", "s3:ListBucket"}
    get = next(s for s in statements if s["Action"] == "s3:GetObject")
    assert "/artifacts/*" in str(get["Resource"])
    listing = next(s for s in statements if s["Action"] == "s3:ListBucket")
    assert listing["Condition"] == {"StringLike": {"s3:prefix": ["artifacts/*"]}}


def test_execution_role_is_separate_and_has_no_s3_access(template: Template) -> None:
    actions = actions_of(policy_statements(template, "ExecutionRole"))
    assert actions
    assert all(a.startswith(("ecr:", "logs:")) for a in actions)


def test_deploy_role_trusts_only_this_repos_deployment_environment(template: Template) -> None:
    template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "AssumeRolePolicyDocument": {
                "Statement": [
                    Match.object_like(
                        {
                            "Action": "sts:AssumeRoleWithWebIdentity",
                            "Condition": {
                                "StringEquals": {
                                    "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                                    "token.actions.githubusercontent.com:sub": (
                                        "repo:aicgoraya/wc2026:environment:production"
                                    ),
                                }
                            },
                        }
                    )
                ]
            }
        },
    )
    statements = policy_statements(template, "DeployRole")
    actions = actions_of(statements)
    assert not {a for a in actions if a.endswith("*")}
    assert not {a for a in actions if a.startswith(("s3:", "iam:Create", "iam:Put", "iam:Attach"))}
    pass_role = next(s for s in statements if s["Action"] == "iam:PassRole")
    assert pass_role["Condition"] == {
        "StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}
    }
    assert len(pass_role["Resource"]) == 2  # exactly the task and execution roles


def test_outputs_cover_every_deploy_workflow_variable(template: Template) -> None:
    outputs = set(template.find_outputs("*"))
    assert outputs == {
        "ServiceUrl",
        "ArtifactBucket",
        "EcrRepository",
        "EcsCluster",
        "EcsService",
        "DeployRoleArn",
        "LogGroup",
    }


def test_running_tasks_require_a_real_image() -> None:
    with pytest.raises(Exception, match="needs an explicit image"):
        synth(desired_count=1)
    running = synth(
        desired_count=1, image="111111111111.dkr.ecr.us-east-1.amazonaws.com/x@sha256:abc"
    )
    running.has_resource_properties("AWS::ECS::Service", {"DesiredCount": 1})
