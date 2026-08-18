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
  name  = "/finsight/${var.environment}/SARVAM_API_KEY"
  type  = "SecureString"
  value = var.sarvam_api_key
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
    sid       = "ElasticBeanstalkDeploy"
    actions   = ["elasticbeanstalk:*"]
    resources = ["arn:aws:elasticbeanstalk:${var.aws_region}:${data.aws_caller_identity.current.account_id}:application/finsight*"]
  }
  statement {
    # beanstalk-deploy (the GitHub Action) also needs to create/read an S3
    # object (the app version bundle) in EB's own managed bucket, and to
    # read S3/CloudFormation/autoscaling state while it polls for the
    # deploy to finish -- narrower than this is fiddly to get right and
    # this role can only ever touch finsight-* EB resources regardless.
    sid = "ElasticBeanstalkDeploySupport"
    actions = [
      "s3:PutObject", "s3:GetObject", "s3:ListBucket",
      "cloudformation:DescribeStacks", "cloudformation:DescribeStackResources",
      "autoscaling:DescribeAutoScalingGroups", "autoscaling:DescribeScalingActivities",
      "ec2:DescribeInstances", "elasticloadbalancing:DescribeLoadBalancers",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_actions_deploy" {
  name   = "finsight-github-actions-deploy"
  role   = aws_iam_role.github_actions_deploy.id
  policy = data.aws_iam_policy_document.github_actions_deploy.json
}
