# Single-Instance EB gets no load balancer (that's the whole point -- an
# ALB/NLB is never free-tier eligible), which means no built-in TLS
# termination either: aws_elastic_beanstalk_environment.backend.cname is
# plain http://. The Vercel frontend is HTTPS, and browsers block an HTTPS
# page from calling a plain-HTTP API (mixed content) -- so this CloudFront
# distribution sits in front of the EB origin purely to add free HTTPS via
# the default *.cloudfront.net certificate, no custom domain/ACM cert
# needed. CloudFront's own free tier (1TB/month data transfer) comfortably
# covers this app's traffic, so this doesn't reintroduce the cost this
# whole Single-Instance choice exists to avoid.
#
# Different origin type than DigiNyaya's cloudfront.tf (that one fronts a
# private S3 bucket of static files with an origin access control; this
# fronts a live dynamic API over plain HTTP) -- caching is fully disabled
# and every header/cookie/query-string/method is forwarded via AWS's own
# managed policies, the standard way to make CloudFront a transparent
# passthrough (this also keeps the /v1/voice/wake-listen WebSocket working,
# not just plain REST calls).

data "aws_cloudfront_cache_policy" "caching_disabled" {
  name = "Managed-CachingDisabled"
}

data "aws_cloudfront_origin_request_policy" "all_viewer" {
  name = "Managed-AllViewer"
}

resource "aws_cloudfront_distribution" "backend" {
  enabled = true
  comment = "FinSight backend HTTPS passthrough (${var.environment})"

  origin {
    domain_name = aws_elastic_beanstalk_environment.backend.cname
    origin_id   = "finsight-backend-eb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only" # the EB origin itself only speaks HTTP -- see this file's own header comment
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    allowed_methods          = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods           = ["GET", "HEAD"]
    target_origin_id         = "finsight-backend-eb"
    viewer_protocol_policy   = "redirect-to-https"
    compress                 = true
    cache_policy_id          = data.aws_cloudfront_cache_policy.caching_disabled.id
    origin_request_policy_id = data.aws_cloudfront_origin_request_policy.all_viewer.id
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}
