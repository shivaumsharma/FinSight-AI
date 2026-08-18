output "backend_url" {
  description = "CloudFront HTTPS URL in front of the EB environment -- this is the FINSIGHT_API_URL value to set on Vercel (the plain EB URL itself is HTTP-only and would get blocked by browsers as mixed content)."
  value       = "https://${aws_cloudfront_distribution.backend.domain_name}"
}

output "backend_eb_url_http_only" {
  description = "The raw EB URL, plain HTTP -- for direct debugging/curl only, never give this to the frontend."
  value       = "http://${aws_elastic_beanstalk_environment.backend.cname}"
}

output "ecr_repository_url" {
  value = aws_ecr_repository.backend.repository_url
}

output "github_actions_role_arn" {
  description = "Add as the AWS_DEPLOY_ROLE_ARN GitHub Actions repo variable."
  value       = aws_iam_role.github_actions_deploy.arn
}

output "eb_application_name" {
  value = aws_elastic_beanstalk_application.finsight.name
}

output "eb_environment_name" {
  value = aws_elastic_beanstalk_environment.backend.name
}
