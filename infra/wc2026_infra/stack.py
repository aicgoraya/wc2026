"""One stack: everything the prediction service needs, and nothing else.

    internet -> ALB (public subnets) -> Fargate task :8080 (public subnets,
                                         ingress only from the ALB's security group)
    task -> S3 (pinned model artifact, read-only)   task stdout -> CloudWatch Logs
    GitHub Actions (OIDC) -> deploy role -> ECR push + ECS task/service update

Networking choice (documented in the README): two public subnets and NO NAT
gateway. The task gets a public IP only so it can reach ECR, S3 and CloudWatch
without paying for a NAT gateway or interface endpoints; its security group
accepts traffic from the load balancer alone, so it is not reachable from the
internet directly.
"""

import dataclasses

import aws_cdk as cdk
from aws_cdk import aws_certificatemanager as acm
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_elasticloadbalancingv2 as elbv2
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from constructs import Construct

CONTAINER_PORT = 8080
GITHUB_OIDC_URL = "https://token.actions.githubusercontent.com"
BOOTSTRAP_TAG = "bootstrap"


@dataclasses.dataclass(frozen=True)
class ServiceConfig:
    """Deploy-time parameters (CDK context)."""

    env_name: str = "demo"
    github_repo: str = "aicgoraya/wc2026"
    github_environment: str = "production"
    artifact_version: str = "unset"
    artifact_prefix: str = "artifacts"
    image: str | None = None
    """Full image reference for the task definition. Unset on first deploy: the
    task definition then points at a placeholder tag and the service starts at
    zero tasks until the deploy workflow publishes a real image."""
    desired_count: int = 0
    cpu: int = 256
    memory_mib: int = 512
    log_retention_days: int = 30
    certificate_arn: str | None = None
    create_oidc_provider: bool = True


class Wc2026ServiceStack(cdk.Stack):
    """ECR + S3 + ECS Fargate behind an ALB + CloudWatch logs + deploy role."""

    def __init__(
        self, scope: Construct, construct_id: str, *, config: ServiceConfig, **kwargs: object
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)  # type: ignore[arg-type]
        name = f"wc2026-{config.env_name}"
        if config.desired_count > 0 and not config.image:
            raise ValueError(
                "desired_count > 0 needs an explicit image (-c image=<registry>/<repo>@sha256:...);"
                " without one the task definition points at a placeholder tag that does not exist"
            )

        # --- storage -------------------------------------------------------
        bucket = s3.Bucket(
            self,
            "Artifacts",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            object_ownership=s3.ObjectOwnership.BUCKET_OWNER_ENFORCED,
            removal_policy=cdk.RemovalPolicy.RETAIN,  # model artifacts survive `cdk destroy`
        )
        repository = ecr.Repository(
            self,
            "Repository",
            repository_name=f"{name}-api",
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            image_scan_on_push=True,
            lifecycle_rules=[
                ecr.LifecycleRule(max_image_count=10, description="keep the last 10 images")
            ],
            removal_policy=cdk.RemovalPolicy.DESTROY,  # images are rebuildable from git
            empty_on_delete=True,
        )
        log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/ecs/{name}-api",
            retention=_retention(config.log_retention_days),
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )

        # --- network -------------------------------------------------------
        vpc = ec2.Vpc(
            self,
            "Vpc",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                )
            ],
        )
        alb_sg = ec2.SecurityGroup(
            self,
            "AlbSg",
            vpc=vpc,
            description="public HTTP(S) to the load balancer",
            allow_all_outbound=False,
        )
        task_sg = ec2.SecurityGroup(
            self,
            "TaskSg",
            vpc=vpc,
            description="API tasks: ingress from the ALB only",
            allow_all_outbound=True,
        )
        task_sg.add_ingress_rule(alb_sg, ec2.Port.tcp(CONTAINER_PORT), "ALB to API")
        alb_sg.add_egress_rule(task_sg, ec2.Port.tcp(CONTAINER_PORT), "ALB to API")

        # --- roles ---------------------------------------------------------
        # Application permissions: read the artifact prefix, nothing else.
        task_role = iam.Role(
            self,
            "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="wc2026 API application role",
        )
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:GetObject"],
                resources=[bucket.arn_for_objects(f"{config.artifact_prefix}/*")],
            )
        )
        # ListBucket (prefix-scoped) only so a missing version reads as 404, not 403.
        task_role.add_to_policy(
            iam.PolicyStatement(
                actions=["s3:ListBucket"],
                resources=[bucket.bucket_arn],
                conditions={"StringLike": {"s3:prefix": [f"{config.artifact_prefix}/*"]}},
            )
        )
        # Execution permissions (pull image, write logs) are separate and granted by CDK below.
        execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            description="wc2026 API ECS agent role",
        )

        # --- compute -------------------------------------------------------
        cluster = ecs.Cluster(self, "Cluster", vpc=vpc, cluster_name=name)
        task_definition = ecs.FargateTaskDefinition(
            self,
            "TaskDef",
            family=f"{name}-api",
            cpu=config.cpu,
            memory_limit_mib=config.memory_mib,
            task_role=task_role,
            execution_role=execution_role,
            runtime_platform=ecs.RuntimePlatform(
                cpu_architecture=ecs.CpuArchitecture.X86_64,
                operating_system_family=ecs.OperatingSystemFamily.LINUX,
            ),
        )
        image = (
            ecs.ContainerImage.from_registry(config.image)
            if config.image
            else ecs.ContainerImage.from_ecr_repository(repository, BOOTSTRAP_TAG)
        )
        task_definition.add_container(
            "api",
            container_name="api",
            image=image,
            essential=True,
            port_mappings=[ecs.PortMapping(container_port=CONTAINER_PORT)],
            environment={
                "WC2026_ARTIFACT_SOURCE": "s3",
                "WC2026_ARTIFACT_BUCKET": bucket.bucket_name,
                "WC2026_ARTIFACT_PREFIX": config.artifact_prefix,
                "WC2026_ARTIFACT_VERSION": config.artifact_version,
                "WC2026_LOG_LEVEL": "INFO",
            },
            logging=ecs.LogDrivers.aws_logs(stream_prefix="api", log_group=log_group),
            stop_timeout=cdk.Duration.seconds(30),
        )
        repository.grant_pull(execution_role)

        service = ecs.FargateService(
            self,
            "Service",
            service_name=f"{name}-api",
            cluster=cluster,
            task_definition=task_definition,
            desired_count=config.desired_count,
            assign_public_ip=True,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            security_groups=[task_sg],
            min_healthy_percent=100,
            max_healthy_percent=200,
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
            health_check_grace_period=cdk.Duration.seconds(120),
            enable_execute_command=False,
        )

        # --- ingress -------------------------------------------------------
        alb = elbv2.ApplicationLoadBalancer(
            self,
            "Alb",
            vpc=vpc,
            internet_facing=True,
            security_group=alb_sg,
            idle_timeout=cdk.Duration.seconds(60),
            drop_invalid_header_fields=True,
        )
        target_group = elbv2.ApplicationTargetGroup(
            self,
            "Targets",
            vpc=vpc,
            port=CONTAINER_PORT,
            protocol=elbv2.ApplicationProtocol.HTTP,
            target_type=elbv2.TargetType.IP,
            targets=[
                service.load_balancer_target(container_name="api", container_port=CONTAINER_PORT)
            ],
            deregistration_delay=cdk.Duration.seconds(30),
            health_check=elbv2.HealthCheck(
                path="/ready",
                healthy_http_codes="200",
                interval=cdk.Duration.seconds(15),
                timeout=cdk.Duration.seconds(5),
                healthy_threshold_count=2,
                unhealthy_threshold_count=3,
            ),
        )
        if config.certificate_arn:
            certificate = acm.Certificate.from_certificate_arn(self, "Cert", config.certificate_arn)
            alb.add_listener(
                "Https",
                port=443,
                certificates=[certificate],
                ssl_policy=elbv2.SslPolicy.RECOMMENDED_TLS,
                default_target_groups=[target_group],
                open=True,
            )
            alb.add_listener(
                "Http",
                port=80,
                default_action=elbv2.ListenerAction.redirect(
                    protocol="HTTPS", port="443", permanent=True
                ),
                open=True,
            )
            scheme = "https"
        else:
            # HTTP only: a demo endpoint, NOT production-ready. Pass certificate_arn for TLS.
            alb.add_listener("Http", port=80, default_target_groups=[target_group], open=True)
            scheme = "http"

        # --- CI/CD: GitHub OIDC deploy role ---------------------------------
        if config.create_oidc_provider:
            # native CloudFormation resource (no Lambda-backed custom resource)
            provider_arn = iam.CfnOIDCProvider(
                self, "GithubOidc", url=GITHUB_OIDC_URL, client_id_list=["sts.amazonaws.com"]
            ).attr_arn
        else:
            provider_arn = self.format_arn(
                service="iam",
                region="",
                resource="oidc-provider",
                resource_name="token.actions.githubusercontent.com",
            )
        deploy_role = iam.Role(
            self,
            "DeployRole",
            description=f"GitHub Actions deploys for {config.github_repo} ({config.github_environment})",
            max_session_duration=cdk.Duration.hours(1),
            assumed_by=iam.FederatedPrincipal(
                provider_arn,
                conditions={
                    "StringEquals": {
                        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
                        # only jobs running in this repo's protected deployment environment
                        "token.actions.githubusercontent.com:sub": f"repo:{config.github_repo}:environment:{config.github_environment}",
                    }
                },
                assume_role_action="sts:AssumeRoleWithWebIdentity",
            ),
        )
        deploy_role.add_to_policy(
            iam.PolicyStatement(actions=["ecr:GetAuthorizationToken"], resources=["*"])
        )
        deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                    "ecr:BatchGetImage",
                    "ecr:DescribeImages",
                ],
                resources=[repository.repository_arn],
            )
        )
        deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:DescribeServices", "ecs:UpdateService"],
                resources=[service.service_arn],
            )
        )
        # these two ECS actions do not support resource-level permissions
        deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=["ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition"],
                resources=["*"],
            )
        )
        deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=["iam:PassRole"],
                resources=[task_role.role_arn, execution_role.role_arn],
                conditions={"StringEquals": {"iam:PassedToService": "ecs-tasks.amazonaws.com"}},
            )
        )

        # --- outputs (these are the GitHub environment variables) -----------
        outputs = {
            "ServiceUrl": f"{scheme}://{alb.load_balancer_dns_name}",
            "ArtifactBucket": bucket.bucket_name,
            "EcrRepository": repository.repository_name,
            "EcsCluster": cluster.cluster_name,
            "EcsService": service.service_name,
            "DeployRoleArn": deploy_role.role_arn,
            "LogGroup": log_group.log_group_name,
        }
        for key, value in outputs.items():
            cdk.CfnOutput(self, key, value=value)


def _retention(days: int) -> logs.RetentionDays:
    allowed = {
        7: logs.RetentionDays.ONE_WEEK,
        14: logs.RetentionDays.TWO_WEEKS,
        30: logs.RetentionDays.ONE_MONTH,
        60: logs.RetentionDays.TWO_MONTHS,
        90: logs.RetentionDays.THREE_MONTHS,
    }
    if days not in allowed:
        raise ValueError(f"log_retention_days must be one of {sorted(allowed)}, got {days}")
    return allowed[days]
