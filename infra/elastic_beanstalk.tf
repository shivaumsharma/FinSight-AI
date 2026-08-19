data "aws_iam_policy_document" "eb_service_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["elasticbeanstalk.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "eb_service" {
  name               = "finsight-eb-service-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.eb_service_assume_role.json
}

resource "aws_iam_role_policy_attachment" "eb_service" {
  role       = aws_iam_role.eb_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSElasticBeanstalkEnhancedHealth"
}

# Confirmed live: an environment update failed with "You do not have
# permission to perform the 'ec2:DescribeSubnets' action" -- that's EB's
# OWN service role acting on the environment's behalf (not the GitHub
# Actions deploy role, which is a separate IAM identity), managing the
# underlying CloudFormation stack's EC2/networking resources. AWS's own
# docs pair this with AWSElasticBeanstalkEnhancedHealth for exactly this
# reason; only the health policy was attached before.
resource "aws_iam_role_policy_attachment" "eb_service_core" {
  role       = aws_iam_role.eb_service.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSElasticBeanstalkService"
}

resource "aws_elastic_beanstalk_application" "finsight" {
  name        = "finsight"
  description = "FinSight AI backend (FastAPI, single-container Docker)"
}

resource "aws_elastic_beanstalk_environment" "backend" {
  name                = "finsight-backend-${var.environment}"
  application         = aws_elastic_beanstalk_application.finsight.name
  solution_stack_name = "64bit Amazon Linux 2023 v4.13.6 running Docker" # confirmed current via `aws elasticbeanstalk list-available-solution-stacks --region eu-north-1`

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "IamInstanceProfile"
    value     = aws_iam_instance_profile.eb_instance.name
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "InstanceType"
    value     = var.eb_instance_type
  }

  setting {
    namespace = "aws:autoscaling:launchconfiguration"
    name      = "SecurityGroups"
    value     = aws_security_group.backend.id
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "EnvironmentType"
    value     = "SingleInstance" # no load balancer -- ALB/NLB are never free-tier eligible; this is what keeps this deployment actually free
  }

  setting {
    namespace = "aws:elasticbeanstalk:environment"
    name      = "ServiceRole"
    value     = aws_iam_role.eb_service.name
  }

  setting {
    namespace = "aws:ec2:vpc"
    name      = "VPCId"
    value     = data.aws_vpc.default.id
  }

  setting {
    namespace = "aws:ec2:vpc"
    name      = "Subnets"
    value     = join(",", data.aws_subnets.default.ids)
  }

  # --- secrets, read from SSM Parameter Store (iam.tf) and injected as real
  # container env vars at deploy time -- never baked into the image, never
  # committed to the repo ---

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "FINNHUB_API_KEY"
    value     = aws_ssm_parameter.finnhub_api_key.value
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "LLM_API_KEY"
    value     = aws_ssm_parameter.llm_api_key.value
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "API_KEY"
    value     = aws_ssm_parameter.api_key.value
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "VAPID_PRIVATE_KEY"
    # EB's Docker platform packs every env var into ONE combined
    # comma-separated "EnvironmentVariables" CloudFormation parameter
    # internally -- a value containing a REAL newline breaks its parsing
    # of that combined list. Re-escaping back to literal "\n" here (HCL
    # already turned the tfvars value's "\n" into a real newline by this
    # point) survives that, and app/api/auth.py's own
    # VAPID_PRIVATE_KEY_PEM.replace("\\n", "\n") already exists
    # specifically to unescape exactly this shape at runtime -- same
    # "most platform env-var UIs can't hold a real multi-line value"
    # reasoning that comment already documents, this EB platform is just
    # another instance of it.
    value = replace(aws_ssm_parameter.vapid_private_key.value, "\n", "\\n")
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "PDF_SHARE_SECRET"
    value     = aws_ssm_parameter.pdf_share_secret.value
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "SARVAM_API_KEY"
    value     = aws_ssm_parameter.sarvam_api_key.value
  }

  # --- non-secret config ---

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "LLM_PROVIDER"
    value     = "hosted"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "LLM_BASE_URL"
    value     = "https://api.groq.com/openai/v1"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "LLM_MODEL"
    value     = var.llm_model
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "MAX_CONCURRENT_JOBS"
    value     = "1"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "LOG_LLM_CALLS"
    value     = "true"
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "VAPID_CONTACT_EMAIL"
    value     = var.vapid_contact_email
  }

  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "ALLOWED_ORIGINS"
    value     = var.frontend_url
  }

  # app/core/paths.py's own persistence mechanism (originally built for
  # Railway's volume model) -- points jobs.db/reports/logs at a directory
  # on the HOST instance's disk (see backend/Dockerrun.aws.json's Volumes
  # mapping in the repo root), not inside the container's own writable
  # layer. Survives an ordinary app-version deploy (a new container just
  # gets started against the same host directory); does NOT survive an
  # actual instance replacement (env rebuild, certain config changes that
  # force one) -- an accepted tradeoff at this traffic/cost tier, same
  # spirit as Railway's own single-volume model, not the stronger RDS-based
  # durability DigiNyaya moved to for its Postgres data.
  setting {
    namespace = "aws:elasticbeanstalk:application:environment"
    name      = "DATA_DIR"
    value     = "/data"
  }
}
