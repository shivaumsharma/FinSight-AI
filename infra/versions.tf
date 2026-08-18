terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Reuses DigiNyaya's already-bootstrapped state bucket (same AWS account,
  # 753779157603) under a different key -- one bucket safely holds multiple
  # projects' state as long as each uses its own key, so this doesn't need
  # its own one-time bootstrap (see DigiNyaya's infra/README.md for how that
  # bucket was created, if it ever needs recreating).
  backend "s3" {
    bucket       = "diginyaya-terraform-state-753779157603"
    key          = "finsight/terraform.tfstate"
    region       = "eu-north-1"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project   = "finsight-ai"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}
