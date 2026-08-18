# FinSight AWS infrastructure (Terraform)

Provisions an Elastic Beanstalk Single-Instance environment (Docker, `t3.micro`)
for the backend, a CloudFront distribution in front of it for free HTTPS, an
ECR repo for the image, and the IAM roles/security group tying it together.
All in `eu-north-1` (Stockholm) -- same region and same AWS account as
DigiNyaya, sharing its already-bootstrapped Terraform state bucket under a
different key.

**Nothing here has been applied.** This is reviewable code -- read the plan
(`terraform plan`) before ever running `terraform apply`, and treat every
`apply` as a real action even though it's sized to cost $0.

## Why this exists

Google Cloud Run (the current production backend) has billing disabled on
that project -- no payment method attached, so the live service is down.
This stands up an equivalent on AWS instead, deliberately sized to stay
inside AWS's free tier rather than draw down the AWS credit balance, since
the whole point was avoiding another cloud bill.

## No bootstrap needed

Unlike a from-scratch AWS project, the Terraform state bucket
(`diginyaya-terraform-state-753779157603`) already exists from DigiNyaya's
own setup -- this just uses a different `key` inside it
(`finsight/terraform.tfstate`), so there's no one-time bucket-creation step
here. `aws configure` still needs to be set up locally with real credentials
if it isn't already.

## Steps

1. Copy `terraform.tfvars.example` to `terraform.tfvars` and fill in real
   values. Two of them (`vapid_private_key`, `pdf_share_secret`) must be the
   **existing** values from the current Cloud Run deployment, not new ones --
   reusing them keeps existing push-notification subscriptions and PDF share
   links working; generating fresh ones breaks every existing one silently.

2. ```bash
   cd infra
   terraform init
   terraform plan     # review every resource before applying anything
   terraform apply
   ```

3. `terraform output` gives you `github_actions_role_arn`,
   `eb_application_name`, `eb_environment_name`, and `backend_url`. Set the
   first three as GitHub Actions repo variables (`AWS_DEPLOY_ROLE_ARN`,
   `EB_APPLICATION_NAME`, `EB_ENVIRONMENT_NAME`) -- `.github/workflows/deploy-aws.yml`
   is already wired to use them and no-ops until `AWS_DEPLOY_ROLE_ARN` is set.

4. Push to `main` (or re-run the workflow manually) to trigger the first
   real deploy -- builds the backend image, pushes to ECR, deploys to EB.

5. Set `FINSIGHT_API_URL` on Vercel to `terraform output backend_url` (the
   `https://...cloudfront.net` one, **not** the plain `http://...elasticbeanstalk.com`
   one -- browsers block an HTTPS page calling a plain-HTTP API).

## What this does NOT do

- Does not touch DNS/a custom domain -- reachable via CloudFront's own
  `*.cloudfront.net` URL, same as DigiNyaya's frontend until a real domain
  is wired up.
- Does not delete or touch anything on Google Cloud -- that project's
  billing is already disabled, so nothing there is actively costing
  anything either; this doesn't need to clean it up to be safe.
- Does not use RDS/Postgres -- FinSight's SQLite file lives on the EC2
  instance's own disk (see `Dockerrun.aws.json`'s `Volumes` mapping and
  `elastic_beanstalk.tf`'s `DATA_DIR` setting), which survives an ordinary
  app-version deploy but NOT an actual instance replacement (env rebuild).
  A real durability upgrade (matching DigiNyaya's move to RDS) is a later
  step if this ever matters enough to justify the added cost/complexity --
  not needed for personal-scale use today.

## Cost -- and one real risk worth reading

Sized to actually be free, not just cheap: EB's `t3.micro` (750 free
hours/month covers running 24/7) + CloudFront's free 1TB/month + ECR's free
tier storage. No load balancer (never free-tier eligible on any account) --
that's the whole reason this is Single-Instance + CloudFront instead of a
load-balanced environment or ECS/Fargate. Verify free-tier eligibility is
still active on the account before applying (`aws freetier get-free-tier-usage`);
if it's expired, `t3.micro` still costs only a few dollars/month.

**The real risk**: `t3.micro` has 1GB of RAM. FinSight's own Dockerfile
notes that torch/transformers/chromadb (FinBERT sentiment + RAG retrieval)
load regardless of `LLM_PROVIDER` -- a real research job invoking all of
that may not fit in 1GB and could get OOM-killed. Chat-only usage (the
conversational trading agent, portfolio/watchlist browsing) doesn't touch
that heavy stack and should be fine. If full research reports crash in
practice, bump `eb_instance_type` to `t3.small` in `terraform.tfvars` and
re-apply -- costs roughly $15/month at that point, no longer free, but a
one-variable change, not a rebuild.
