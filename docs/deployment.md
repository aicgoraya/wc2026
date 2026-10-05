# Deploying the prediction service to AWS

Nothing in this document has been executed against a real AWS account yet. The
infrastructure code synthesizes and its template assertions pass; the workflow
is linted. Treat the first run as a first run.

All commands assume the repository root unless noted, `make setup` done, Node.js
22+, and AWS credentials for the target account in your shell (SSO, a profile,
whatever you normally use). No long-lived keys are stored anywhere in this
setup.

## What gets created

One CloudFormation stack, `wc2026-<env_name>` (default `wc2026-demo`):

| Resource | Notes |
|---|---|
| VPC, 2 public subnets, internet gateway | No NAT gateway, no private subnets |
| Application Load Balancer + target group + listener | Internet-facing, HTTP 80 (HTTPS 443 + redirect if you pass a certificate). Health check: `GET /ready` |
| 2 security groups | ALB: 80/443 from anywhere. Task: 8080 from the ALB group only |
| ECS cluster, Fargate task definition, service | 0.25 vCPU / 512 MB, x86-64, starts at **0 tasks**, deployment circuit breaker with rollback |
| ECR repository `wc2026-<env>-api` | Immutable tags, scan on push, keeps the last 10 images |
| S3 bucket | Private, public access blocked, encrypted, versioned, TLS-only, **retained on destroy** |
| CloudWatch log group `/ecs/wc2026-<env>-api` | 30-day retention |
| IAM: execution role, application role, deploy role | See "Permissions" |
| IAM OIDC provider for GitHub | Skipped with `-c create_oidc_provider=false` if the account already has one |

Tags on everything: `project=wc2026`, `environment=<env_name>`.

### Cost drivers (no prices quoted; check the AWS pricing pages for your region)

- **Load balancer**: billed per hour it exists plus usage. The largest fixed cost, and it bills
  even with zero tasks.
- **Fargate task**: billed per second for 0.25 vCPU and 0.5 GB while a task runs.
- **Public IPv4 addresses**: two for the load balancer, one for the task, billed hourly.
- **CloudWatch Logs**: ingestion and storage. One line per request, including the load
  balancer's health checks (every 15 s from each of two zones).
- **ECR and S3 storage**: up to 10 images of a few hundred MB; artifacts are about 3 MB each.
- **Data transfer out** for responses.

To pause the task without destroying anything:
`aws ecs update-service --cluster <EcsCluster> --service <EcsService> --desired-count 0`.
The load balancer and its addresses keep billing until the stack is destroyed.

### Context parameters

Pass with `-c key=value` to any `cdk` command.

| Key | Default | Meaning |
|---|---|---|
| `env_name` | `demo` | Suffix for names and the `environment` tag |
| `artifact_version` | `unset` | Model artifact the task pins (`WC2026_ARTIFACT_VERSION`) |
| `artifact_prefix` | `artifacts` | S3 key prefix |
| `image` | none | Full image reference. Leave unset for the first deploy |
| `desired_count` | `0` | Tasks. More than 0 requires `image` |
| `cpu`, `memory_mib` | `256`, `512` | Task size |
| `log_retention_days` | `30` | One of 7, 14, 30, 60, 90 |
| `certificate_arn` | none | ACM certificate in the same region → HTTPS |
| `github_repo`, `github_environment` | `aicgoraya/wc2026`, `production` | Who may assume the deploy role |
| `create_oidc_provider` | `true` | `false` if the account already has the GitHub OIDC provider |
| `region` | from your credentials | Deployment region |

## First deployment

```bash
cd infra
CDK="uv run --group infra npx --yes aws-cdk@2.1144.0"

# 0. once per account/region: CDK's own bootstrap stack (a small S3 bucket + roles)
$CDK bootstrap

# 1. read the plan before creating anything
$CDK diff -c artifact_version=2026-10-05.1

# 2. create the stack. The service has 0 tasks, so nothing runs yet.
$CDK deploy -c artifact_version=2026-10-05.1
#    note the outputs: ServiceUrl, ArtifactBucket, EcrRepository, EcsCluster,
#    EcsService, DeployRoleArn, LogGroup

# 3. publish the model artifact
cd ..
uv run wc2026 upload-artifact --bundle artifacts/2026-10-05.1 --bucket <ArtifactBucket>
```

4. In GitHub: **Settings → Environments → New environment → `production`**.
   - Add these *environment variables* (not secrets; none of them is sensitive):

     | Variable | Value |
     |---|---|
     | `AWS_REGION` | the region you deployed to |
     | `AWS_DEPLOY_ROLE_ARN` | `DeployRoleArn` output |
     | `ECR_REPOSITORY` | `EcrRepository` output |
     | `ECS_CLUSTER` | `EcsCluster` output |
     | `ECS_SERVICE` | `EcsService` output |
     | `SERVICE_URL` | `ServiceUrl` output (or your own HTTPS domain) |

   - Recommended: restrict the environment's deployment branches to `main` and add
     yourself as a required reviewer. The AWS role trusts only
     `repo:aicgoraya/wc2026:environment:production`, so no other branch, fork or
     pull request can obtain AWS credentials.
5. Merge this code to `main`, then **Actions → deploy → Run workflow** (leave
   `artifact_version` blank to keep the pinned one). The workflow:
   1. re-runs the CI checks on that exact commit;
   2. builds the image and smoke-tests that image with the synthetic artifact;
   3. pushes it to ECR as `<commit sha>` (immutable) and resolves its digest;
   4. registers a new task-definition revision that references the image **by digest**;
   5. updates the service to 1 task and waits (at most 15 minutes) for it to stabilise,
      then confirms the service is stable *on the new revision*;
   6. runs `tools/smoke.py` against `SERVICE_URL`, requiring the expected real artifact;
   7. on failure after step 5 began, restores the previous task definition and task count.

A deployment has succeeded only when that run is green. Until then, do not
describe the service as deployed.

### HTTPS

Request or import an ACM certificate for a domain you control (same region),
deploy with `-c certificate_arn=<arn>`, point a DNS record at the load balancer,
and set `SERVICE_URL` to `https://<your domain>`. Without this the endpoint is
unencrypted HTTP and should be treated as a demo.

### Changing infrastructure later

The deploy workflow registers task-definition revisions outside CloudFormation.
A later `cdk deploy` that changes the task definition (CPU, memory, environment,
artifact version) replaces it with CDK's version, which points at a placeholder
image and 0 tasks unless you tell it otherwise. Either pass the current image
and count, `-c image=<registry>/<repo>@sha256:… -c desired_count=1` (the image
reference is in the last deploy run's summary), or accept a brief scale-to-zero
and re-run the deploy workflow afterwards.

## Shipping a new model

```bash
uv run wc2026 ingest-history
uv run wc2026 export-artifact --version <new-version>
uv run wc2026 upload-artifact --bundle artifacts/<new-version> --bucket <ArtifactBucket>
```

Then run the deploy workflow with `artifact_version=<new-version>`. The service
never picks up "the latest" object on its own, and an existing version cannot be
overwritten by `upload-artifact`.

## Logs and troubleshooting

```bash
aws logs tail /ecs/wc2026-demo-api --follow            # live
aws logs tail /ecs/wc2026-demo-api --since 1h --filter-pattern '{ $.level = "ERROR" }'
```

CloudWatch Logs Insights (select the log group; the JSON fields are discovered automatically).
These queries use standard Insights syntax but have not been run against a live log group yet.

```
# latency and volume per route, 5-minute buckets
filter event = "request"
| stats count(*) as requests, pct(duration_ms, 50) as p50_ms, pct(duration_ms, 95) as p95_ms by route, bin(5m)

# server errors, newest first (use request_id to find the traceback line)
fields @timestamp, route, status, error_type, request_id
| filter event = "request" and status >= 500
| sort @timestamp desc

# client errors by cause
filter event = "request" and status >= 400 and status < 500
| stats count(*) by error_code, route
```

Error rate over a window = the second query's count divided by the first's.
No alarms are defined: with one demo task and no traffic baseline there is no
threshold I could justify.

**The task will not become healthy.** Look for the startup lines:

| Log line / symptom | Meaning | Fix |
|---|---|---|
| `artifact_load_failed`, `reason: artifact_not_found` | No `manifest.json` at `s3://<bucket>/<prefix>/<version>/` | Upload the artifact, or deploy with the right `artifact_version` |
| `reason: artifact_store_unavailable` | Access denied or S3 unreachable (`detail` says which) | Check the application role and the bucket name |
| `reason: artifact_corrupt` | A file does not match its SHA-256, or is malformed | Re-export under a new version and upload again |
| `reason: artifact_incompatible` | Feature-schema, manifest or LightGBM major version differs from the image | Deploy an image and artifact built from the same code |
| `reason: artifact_config_invalid` | Missing `WC2026_ARTIFACT_*` variables | Check the task definition's environment |
| No log lines at all | The container never started | `aws ecs describe-services --cluster <c> --services <s> --query 'services[0].events[:10]'` (image pull errors show here) |

`/health` returning 200 while `/ready` returns 503 is the designed "alive but no
model" state; `curl <ServiceUrl>/ready` shows the same reason code.

**A request failed.** Every response carries `x-request-id`; search the log
group for it. A 500 has a single `request` line with `error_type` and the full
traceback.

## Rollback

Three mechanisms. None has been exercised against AWS yet.

| Mechanism | Automatic? | When it acts |
|---|---|---|
| ECS deployment circuit breaker | Automatic | New tasks never pass the `/ready` health check: ECS returns the service to the last completed deployment. The workflow notices the service is not on the new revision and fails |
| Workflow rollback step | Automatic | The rollout check or the post-deployment smoke check fails: the workflow re-points the service at the task definition and task count it recorded before the update |
| Manual | Manual | Anything else, for example a bad model that passes the smoke checks |

Manual rollback to a previous revision (each deploy run's summary lists both the new and the
previous task definition):

```bash
aws ecs list-task-definitions --family-prefix wc2026-demo-api --sort DESC --max-items 5
aws ecs update-service --cluster <EcsCluster> --service <EcsService> \
  --task-definition wc2026-demo-api:<previous-revision>
aws ecs wait services-stable --cluster <EcsCluster> --services <EcsService>
python3 tools/smoke.py <ServiceUrl>
```

To roll back only the model, run the deploy workflow with the previous
`artifact_version`. Rollback targets depend on their image still existing: ECR
keeps the last 10.

## Teardown

```bash
cd infra
uv run --group infra npx --yes aws-cdk@2.1144.0 destroy
```

- **Deleted**: the load balancer, service, cluster, task definitions created by the stack, VPC,
  security groups, IAM roles, the OIDC provider (if this stack created it; other stacks relying on
  it would break), the log group **and its logs**, the ECR repository **and its images**.
- **Kept**: the S3 artifact bucket and every object version in it. To remove it, empty it
  (including versions) in the S3 console and delete it.
- **Also left behind**: task-definition revisions registered by the deploy workflow (inactive,
  free), and CDK's bootstrap stack `CDKToolkit` (delete it separately if nothing else uses CDK in
  that account).

After teardown, clear the variables on the GitHub `production` environment so
the deploy workflow fails fast instead of targeting resources that no longer
exist.

## Permissions

| Role | Trusted by | Can do |
|---|---|---|
| Execution role | ECS tasks service | Pull from this ECR repository; write to this log group |
| Application (task) role | ECS tasks service | `s3:GetObject` on `<bucket>/artifacts/*`; `s3:ListBucket` restricted to that prefix (so a missing version reads as "not found" rather than "access denied") |
| Deploy role | GitHub OIDC, only `repo:aicgoraya/wc2026:environment:production` | Push/describe images in this repository; register/describe task definitions; describe/update this service; pass the two roles above to ECS. No S3, no IAM changes, no CloudFormation |

Creating or destroying the stack is done by you with your own credentials, not
by the deploy role.

## Workflow notes

- Actions are referenced by major version tag (`@v4`, `@v5`, `@v2`), matching the existing
  workflows in this repository. Pinning them to commit SHAs is a reasonable hardening step.
- `deploy` runs only from `main`, only by manual dispatch, and never on pull requests.
- Deployments share one concurrency group and are queued, never run in parallel or cancelled
  mid-rollout.
- Re-running a deploy for a commit whose image is already in ECR reuses that image.
