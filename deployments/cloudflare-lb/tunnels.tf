# --- New CN-side tunnel ---
# Run on the CN host once created:
#   cloudflared tunnel run --token <token from `cloudflared tunnel token` for this tunnel_id>
resource "cloudflare_zero_trust_tunnel_cloudflared" "cn" {
  account_id    = var.account_id
  name          = "embedding-provider-preseen-ai-cn"
  config_src    = "cloudflare"
  tunnel_secret = var.cn_tunnel_secret
}

# --- Virtual networks: one per region so LB origins in different pools never collide ---
resource "cloudflare_zero_trust_tunnel_cloudflared_virtual_network" "us" {
  account_id = var.account_id
  name       = "vnet-embed-us"
  comment    = "Private network for embed.preseen.ai US origin (gpu4)"
}

resource "cloudflare_zero_trust_tunnel_cloudflared_virtual_network" "cn" {
  account_id = var.account_id
  name       = "vnet-embed-cn"
  comment    = "Private network for embed.preseen.ai CN origin"
}

# --- Private IP routes: bind each virtual network to the tunnel that can reach that origin ---
# NOTE: this does not touch the existing public hostname ingress on the US tunnel -- a tunnel can
# carry both a public hostname route and a private IP route at the same time.
resource "cloudflare_zero_trust_tunnel_cloudflared_route" "us" {
  account_id          = var.account_id
  tunnel_id           = var.us_tunnel_id
  network             = "${var.us_origin_address}/32"
  virtual_network_id  = cloudflare_zero_trust_tunnel_cloudflared_virtual_network.us.id
  comment             = "embed.preseen.ai US origin (gpu4)"
}

resource "cloudflare_zero_trust_tunnel_cloudflared_route" "cn" {
  account_id          = var.account_id
  tunnel_id           = cloudflare_zero_trust_tunnel_cloudflared.cn.id
  network             = "${var.cn_origin_address}/32"
  virtual_network_id  = cloudflare_zero_trust_tunnel_cloudflared_virtual_network.cn.id
  comment             = "embed.preseen.ai CN origin"
}
