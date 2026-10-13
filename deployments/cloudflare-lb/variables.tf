variable "account_id" {
  description = "Cloudflare account ID (Zero Trust / Load Balancing objects live at the account level)."
  type        = string
}

variable "zone_id" {
  description = "Zone ID for preseen.ai."
  type        = string
}

variable "hostname" {
  description = "Public hostname the load balancer answers for."
  type        = string
  default     = "embed.preseen.ai"
}

# --- US side (existing production tunnel on gpu4) ---
# This tunnel already exists (created via `cloudflared tunnel create embedding-provider-preseen-ai`
# per README.md). We do NOT recreate it here -- only attach a virtual network + private route to it,
# so the existing public hostname ingress on this tunnel keeps working untouched.
variable "us_tunnel_id" {
  description = "UUID of the existing 'embedding-provider-preseen-ai' tunnel on gpu4 (from `cloudflared tunnel list`)."
  type        = string
}

variable "us_origin_address" {
  description = "Address the US tunnel connector can reach the service on (matches PUBLIC_UPSTREAM in deployments/gpu4/*.env)."
  type        = string
  default     = "127.0.0.1"
}

variable "us_origin_port" {
  type    = number
  default = 7997
}

# --- CN side (new tunnel/host, to be created) ---
variable "cn_tunnel_secret" {
  description = "Base64, >=32 byte secret for the new CN-side tunnel. Generate with: openssl rand -base64 32"
  type        = string
  sensitive   = true
}

variable "cn_origin_address" {
  description = "Address the CN tunnel connector can reach the service on, on the CN host."
  type        = string
  default     = "127.0.0.1"
}

variable "cn_origin_port" {
  type    = number
  default = 7997
}

variable "health_check_path" {
  type    = string
  default = "/healthz"
}
