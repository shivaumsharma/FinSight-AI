# Stopgap HTTPS front door while CloudFront (cloudfront.tf) stays blocked
# on this AWS account's own pending verification case -- API Gateway is a
# separate AWS service with no dependency on that same verification, so it
# gives a real https://...execute-api... URL immediately. Same reason
# CloudFront exists (the EB Single-Instance origin only speaks plain HTTP,
# and an HTTPS Vercel frontend can't call it directly -- mixed content),
# just a different AWS-managed TLS terminator in front of the same origin.
# Not meant to replace cloudfront.tf long-term -- once the verification
# case clears, backend_url can go back to the CloudFront output.

resource "aws_apigatewayv2_api" "backend" {
  name          = "finsight-backend-${var.environment}"
  protocol_type = "HTTP"
}

resource "aws_apigatewayv2_integration" "backend" {
  api_id                 = aws_apigatewayv2_api.backend.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  integration_uri        = "http://${aws_elastic_beanstalk_environment.backend.cname}/{proxy}"
  payload_format_version = "1.0"
}

# Separate integration for the root route -- API Gateway requires every
# {proxy}-style placeholder in an integration's URI to be satisfiable by
# ITS route's own path variables. The root route's key ("ANY /") has no
# {proxy+} segment to supply that value from, so it needs its own
# integration whose URI has no {proxy} placeholder at all.
resource "aws_apigatewayv2_integration" "backend_root" {
  api_id                 = aws_apigatewayv2_api.backend.id
  integration_type       = "HTTP_PROXY"
  integration_method     = "ANY"
  integration_uri        = "http://${aws_elastic_beanstalk_environment.backend.cname}/"
  payload_format_version = "1.0"
}

# Two routes, not one -- {proxy+} only matches a NON-empty path segment,
# so a bare request to the API root (e.g. GET / for a health check style
# call) needs its own route or it 404s at the API Gateway layer before
# ever reaching the backend.
resource "aws_apigatewayv2_route" "proxy" {
  api_id    = aws_apigatewayv2_api.backend.id
  route_key = "ANY /{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.backend.id}"
}

resource "aws_apigatewayv2_route" "root" {
  api_id    = aws_apigatewayv2_api.backend.id
  route_key = "ANY /"
  target    = "integrations/${aws_apigatewayv2_integration.backend_root.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.backend.id
  name        = "$default"
  auto_deploy = true
}
