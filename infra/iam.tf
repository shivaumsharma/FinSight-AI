# --- Secrets, via SSM Parameter Store (SecureString) rather than plaintext
# EB environment variables -- same pattern DigiNyaya's iam.tf already uses ---

resource "aws_ssm_parameter" "finnhub_api_key" {
  name  = "/finsight/${var.environment}/FINNHUB_API_KEY"
  type  = "SecureString"
  value = var.finnhub_api_key
}

resource "aws_ssm_parameter" "llm_api_key" {
  name  = "/finsight/${var.environment}/LLM_API_KEY"
  type  = "SecureString"
  value = var.llm_api_key
}

resource "aws_ssm_parameter" "api_key" {
  name  = "/finsight/${var.environment}/API_KEY"
  type  = "SecureString"
  value = var.api_key
}

resource "aws_ssm_parameter" "vapid_private_key" {
  name  = "/finsight/${var.environment}/VAPID_PRIVATE_KEY"
  type  = "SecureString"
  value = var.vapid_private_key
}

resource "aws_ssm_parameter" "pdf_share_secret" {
  name  = "/finsight/${var.environment}/PDF_SHARE_SECRET"
  type  = "SecureString"
  value = var.pdf_share_secret
}

resource "aws_ssm_parameter" "sarvam_api_key" {
  name = "/finsight/${var.environment}/SARVAM_API_KEY"
  type = "SecureString"
  # SSM rejects an empty-string value outright (ValidationException) --
  # substitute a harmless placeholder when unset, same as the app's own
  # existing "empty/missing SARVAM_API_KEY" degrade path (voice features
  # just fail gracefully either way, see app/api/main.py).
  value = var.sarvam_api_key != "" ? var.sarvam_api_key : "not-configured"
}

# --- EB EC2 instance profile: what the running backend container can do ---

data "aws_iam_policy_document" "eb_instance_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eb_instance" {
  name               = "finsight-eb-instance-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.eb_instance_assume_role.json
}

resource "aws_iam_role_policy_attachment" "eb_instance_ecr" {
  role       = aws_iam_role.eb_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "eb_instance_web_tier" {
  role       = aws_iam_role.eb_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AWSElasticBeanstalkWebTier"
}

data "aws_iam_policy_document" "eb_instance_scoped" {
  statement {
    sid     = "SsmParameterRead"
    actions = ["ssm:GetParameter", "ssm:GetParameters"]
    resources = [
      aws_ssm_parameter.finnhub_api_key.arn,
      aws_ssm_parameter.llm_api_key.arn,
      aws_ssm_parameter.api_key.arn,
      aws_ssm_parameter.vapid_private_key.arn,
      aws_ssm_parameter.pdf_share_secret.arn,
      aws_ssm_parameter.sarvam_api_key.arn,
    ]
  }
  statement {
    # Confirmed live: even after the EB SERVICE role (elastic_beanstalk.tf's
    # eb_service, a separate IAM identity) got AWSElasticBeanstalkService,
    # the exact same "ec2:DescribeSubnets" AccessDenied kept happening --
    # this call is actually made BY the EB host agent running ON the
    # instance itself, using the INSTANCE role, not the service role.
    # Neither AWSElasticBeanstalkWebTier nor
    # AWSElasticBeanstalkMulticontainerDocker (checked both directly)
    # includes it. EC2 Describe* actions don't support resource-level
    # scoping in IAM at all -- "*" is the only valid resource for these,
    # not a shortcut around scoping. A small bundle of the other
    # networking/instance describes the host agent commonly needs
    # alongside it, granted together to avoid a further one-at-a-time
    # hunt. ec2:DescribeLaunchTemplateVersions added after another
    # live AccessDenied on it specifically -- the natural companion to
    # DescribeLaunchTemplates (see github_actions_deploy's own
    # ElasticBeanstalkDeployPolling statement in this same file):
    # Single-Instance EB manages instance replacement via a launch
    # template, and its host agent needs to read both the template AND
    # its versions, not just one.
    sid = "Ec2DescribeForHostAgent"
    actions = [
      "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups", "ec2:DescribeVpcs",
      "ec2:DescribeInstances", "ec2:DescribeAvailabilityZones", "ec2:DescribeTags",
      "ec2:DescribeLaunchTemplates", "ec2:DescribeLaunchTemplateVersions",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "eb_instance_scoped" {
  name   = "finsight-eb-instance-scoped-${var.environment}"
  role   = aws_iam_role.eb_instance.id
  policy = data.aws_iam_policy_document.eb_instance_scoped.json
}

resource "aws_iam_instance_profile" "eb_instance" {
  name = "finsight-eb-instance-${var.environment}"
  role = aws_iam_role.eb_instance.name
}

# --- GitHub Actions OIDC role: what CI can do (ECR push + EB deploy),
# scoped to this one repo and these specific resources -- no long-lived
# access keys. Assumes an OIDC provider for token.actions.githubusercontent.com
# already exists on this AWS account -- it's account-wide, not per-project,
# so if DigiNyaya's infra has already created one (or you created it by hand),
# do NOT create a second one here; this only declares the ROLE that trusts
# it. If neither project has applied yet, create the provider once via:
#   aws iam create-open-id-connect-provider \
#     --url https://token.actions.githubusercontent.com \
#     --client-id-list sts.amazonaws.com \
#     --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1 ---

data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    principals {
      type        = "Federated"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/token.actions.githubusercontent.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:shivaumsharma/FinSight-AI:ref:refs/heads/main"]
    }
  }
}

resource "aws_iam_role" "github_actions_deploy" {
  name               = "finsight-github-actions-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json
}

data "aws_iam_policy_document" "github_actions_deploy" {
  statement {
    sid       = "EcrAuth"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    sid = "EcrPushToOneRepo"
    actions = [
      "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage", "ecr:PutImage", "ecr:InitiateLayerUpload",
      "ecr:UploadLayerPart", "ecr:CompleteLayerUpload",
    ]
    resources = [aws_ecr_repository.backend.arn]
  }
  statement {
    # A real deploy touches several DIFFERENT EB resource *types*, not
    # just "application" -- confirmed live: the first real deploy
    # attempt was denied on CreateApplicationVersion specifically
    # because "application/finsight*" doesn't match an
    # "applicationversion/..." ARN at all (different resource-type
    # segment, not just a naming difference). platform/solutionstack
    # are AWS-managed shared resources (not finsight-specific), needed
    # read-only to resolve the Docker platform this environment runs.
    sid     = "ElasticBeanstalkDeploy"
    actions = ["elasticbeanstalk:*"]
    resources = [
      "arn:aws:elasticbeanstalk:${var.aws_region}:${data.aws_caller_identity.current.account_id}:application/finsight*",
      "arn:aws:elasticbeanstalk:${var.aws_region}:${data.aws_caller_identity.current.account_id}:applicationversion/finsight*",
      "arn:aws:elasticbeanstalk:${var.aws_region}:${data.aws_caller_identity.current.account_id}:environment/finsight*",
      "arn:aws:elasticbeanstalk:${var.aws_region}:${data.aws_caller_identity.current.account_id}:configurationtemplate/finsight*",
      "arn:aws:elasticbeanstalk:${var.aws_region}::platform/*",
      "arn:aws:elasticbeanstalk:${var.aws_region}::solutionstack/*",
    ]
  }
  statement {
    # Full S3 admin, but scoped to ONLY the one bucket EB itself manages
    # (elasticbeanstalk-{region}-{account}), not account-wide -- confirmed
    # live this needs more than just CreateBucket/PutObject: the deploy
    # action also configures the bucket's ownership controls, encryption,
    # versioning, and public-access-block settings on first creation
    # (caught one narrow AccessDenied at a time: CreateBucket, then
    # PutBucketOwnershipControls, ...). Granting the full action set on
    # this single non-sensitive bucket (it only ever holds deployment
    # ZIP/JSON bundles) up front avoids hunting down each one
    # individually -- the blast radius is one bucket, not the account.
    sid = "ElasticBeanstalkDeployBucket"
    actions = ["s3:*"]
    resources = [
      "arn:aws:s3:::elasticbeanstalk-${var.aws_region}-${data.aws_caller_identity.current.account_id}",
      "arn:aws:s3:::elasticbeanstalk-${var.aws_region}-${data.aws_caller_identity.current.account_id}/*",
    ]
  }
  statement {
    # Read-only account/region-wide state the deploy action polls while
    # waiting for the environment update to finish -- these are all
    # Describe*/List*-shape calls, not resource-scoped by AWS's own IAM
    # model for these actions, so "*" here is the actual available
    # granularity, not a shortcut around scoping.
    sid = "ElasticBeanstalkDeployPolling"
    actions = [
      # cloudformation:Describe*/Get*/List* -- broadened past just
      # DescribeStacks/DescribeStackResources after confirmed live that
      # the deploy action also calls GetTemplate while polling an
      # in-progress environment update (EB manages each environment as
      # a CloudFormation stack internally). Read-only, so granting the
      # full read family up front avoids a fourth round of hunting down
      # one more Describe/Get call individually.
      "cloudformation:Describe*", "cloudformation:Get*", "cloudformation:List*",
      "autoscaling:DescribeAutoScalingGroups", "autoscaling:DescribeScalingActivities",
      "ec2:DescribeInstances", "ec2:DescribeLaunchTemplates", "ec2:DescribeLaunchTemplateVersions",
      "elasticloadbalancing:DescribeLoadBalancers",
      # Confirmed live: a real app-version deploy (not just a config/env-var
      # update) failed here with "not authorized to perform:
      # autoscaling:SuspendProcesses" -- EB pauses the underlying ASG's own
      # scaling activity while it replaces the Single-Instance environment's
      # one instance, then resumes it once the new version is healthy.
      # Granted together (SuspendProcesses alone would just trade this
      # error for the same one on ResumeProcesses at the end of the same
      # deploy). Can't be scoped to the specific ASG's ARN in this
      # Terraform config -- that ASG is created dynamically by EB's own
      # internal CloudFormation stack per environment, not by this state.
      "autoscaling:SuspendProcesses", "autoscaling:ResumeProcesses",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "finsight-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
