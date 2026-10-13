terraform {
  required_version = ">= 1.5.0"

  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 5.0"
    }
  }
}

provider "cloudflare" {
  # CLOUDFLARE_API_TOKEN env var, scoped to:
  #   Zone / DNS: Edit  (zone: preseen.ai)
  #   Account / Load Balancing: Monitors and Pools: Edit
  #   Account / Load Balancers: Edit
  #   Account / Cloudflare Tunnel: Edit
  #   Account / Cloudflare One Networks: Edit
}
