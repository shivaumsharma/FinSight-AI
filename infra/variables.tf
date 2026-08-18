variable "aws_region" {
  description = "AWS region for every resource. eu-north-1 to match DigiNyaya (same account, same region keeps cross-project things simple)."
  type        = string
  default     = "eu-north-1"
}

variable "environment" {
  description = "Deployment environment name, used in resource naming."
  type        = string
  default     = "prod"
}

variable "eb_instance_type" {
  description = "EC2 instance type for the Elastic Beanstalk Single-Instance environment. t3.micro specifically -- the free-tier-eligible size (750 hrs/month free for a new-ish account); t3.small (DigiNyaya's size) is NOT free-tier eligible."
  type        = string
  default     = "t3.micro"
}

# --- secrets -- pass via terraform.tfvars (git-ignored) or TF_VAR_* env vars, never commit ---

variable "finnhub_api_key" {
  description = "Finnhub API key (market data)."
  type        = string
  sensitive   = true
}

variable "llm_api_key" {
  description = "Groq (OpenAI-compatible) API key."
  type        = string
  sensitive   = true
}

variable "api_key" {
  description = "X-API-Key value gating this deployment (main.py's require_api_key) -- also what the frontend's FINSIGHT_API_KEY and sweep-cron.yml's FINSIGHT_API_KEY secret must match."
  type        = string
  sensitive   = true
}

variable "vapid_private_key" {
  description = "Web Push VAPID private key (PEM). Generate with scripts/generate_vapid_keys.py -- reuse the SAME keypair already in use if push subscriptions from the current deployment should keep working, otherwise every existing browser subscription breaks."
  type        = string
  sensitive   = true
}

variable "pdf_share_secret" {
  description = "HMAC secret for signed PDF share links (app/api/auth.py's sign_pdf_url). Reuse the current value to keep existing share links valid."
  type        = string
  sensitive   = true
}

variable "sarvam_api_key" {
  description = "Sarvam AI API key (STT/TTS/wake-word). Optional -- voice features degrade gracefully without it (see app/api/main.py), so an empty string is fine if you don't have one."
  type        = string
  sensitive   = true
  default     = ""
}

# --- non-secret config, plain EB environment variables ---

variable "llm_model" {
  description = "Main-pipeline LLM model (app/core/llm_provider.py). NOT chat_router.py's CHAT_MODEL, which is hardcoded in source, not env-configured."
  type        = string
  default     = "openai/gpt-oss-120b"
}

variable "vapid_contact_email" {
  type    = string
  default = "sshivaum@gmail.com"
}

variable "frontend_url" {
  description = "The Vercel frontend's URL, for CORS. Already known up front (unlike DigiNyaya's CloudFront URL, which doesn't exist until after apply) -- no two-phase apply needed here."
  type        = string
  default     = "https://web-ten-blond-39.vercel.app"
}
