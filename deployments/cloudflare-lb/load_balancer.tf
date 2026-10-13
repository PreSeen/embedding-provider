resource "cloudflare_load_balancer_monitor" "embed_health" {
  account_id     = var.account_id
  type           = "https"
  path           = var.health_check_path
  expected_codes = "200"
  method         = "GET"
  # 60s interval is the base-tier rate; 30s/15s each add a flat monthly surcharge
  # on top of the $5 base Load Balancing subscription. Bump this only if you've
  # decided the faster failover is worth the extra cost.
  interval         = 60
  timeout          = 5
  retries          = 2
  consecutive_up   = 1
  consecutive_down = 2
  description      = "embed.preseen.ai /healthz"
}

resource "cloudflare_load_balancer_pool" "us" {
  account_id = var.account_id
  name       = "embed-us"
  monitor    = cloudflare_load_balancer_monitor.embed_health.id
  # Pinned to 1 check region (base-tier). Omitting check_regions checks from
  # every Cloudflare data center, which bills at the highest health-check-region tier.
  check_regions = ["ENAM"]

  origins = [{
    name                = "gpu4"
    address             = var.us_origin_address
    port                = var.us_origin_port
    enabled             = true
    virtual_network_id  = cloudflare_zero_trust_tunnel_cloudflared_virtual_network.us.id
  }]
}

resource "cloudflare_load_balancer_pool" "cn" {
  account_id    = var.account_id
  name          = "embed-cn"
  monitor       = cloudflare_load_balancer_monitor.embed_health.id
  check_regions = ["NEAS"] # North East Asia -- closest available check region to a CN origin

  origins = [{
    name                = "cn-host"
    address             = var.cn_origin_address
    port                = var.cn_origin_port
    enabled             = true
    virtual_network_id  = cloudflare_zero_trust_tunnel_cloudflared_virtual_network.cn.id
  }]
}

resource "cloudflare_load_balancer" "embed" {
  zone_id = var.zone_id
  name    = var.hostname
  proxied = true
  # least_outstanding_requests: Cloudflare's edge tracks how many requests are
  # currently in flight to each pool and sends new requests to whichever pool
  # has fewer pending -- a live read on how busy/backed-up each instance
  # actually is (queueing, batch saturation, GPU offload state per README.md),
  # not network path length. Deliberately NOT dynamic_latency: that policy's
  # RTT comes from the health-monitor probe's network round trip, which for a
  # US/CN pair is dominated by transit distance, not by how loaded either
  # instance is -- it would just silently reproduce geo-style routing by
  # network proximity, which is exactly what we're avoiding here.
  steering_policy = "least_outstanding_requests"

  default_pools = [
    cloudflare_load_balancer_pool.us.id,
    cloudflare_load_balancer_pool.cn.id,
  ]
  fallback_pool = cloudflare_load_balancer_pool.us.id
}
