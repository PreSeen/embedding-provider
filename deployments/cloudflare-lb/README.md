# embed.preseen.ai — US/CN Load Balancer

Terraform module that adds Cloudflare Load Balancing on top of the existing
`embed.preseen.ai` tunnel setup so requests automatically go to whichever of a
US-origin or CN-origin backend instance is currently less loaded
(`steering_policy = "least_outstanding_requests"`), with automatic failover to
the other pool if one goes unhealthy. This intentionally does not use
country-based (`geo`) steering or network-RTT-based (`dynamic_latency`)
steering -- no `country_pools`, no per-country routing, no geo-steering
surcharge, and no routing decision driven by network/geographic distance. The
goal is explicitly to route by how busy each *instance* is (in-flight request
count, a proxy for queueing/batch saturation per the main `README.md`'s
adaptive-batching behavior), not by where the client or server is located.

It does **not** replace the existing tunnel described in the main `README.md`
Remote Sync section (`embedding-provider-preseen-ai` on gpu4) -- it attaches a
private-network route to that tunnel and adds a second tunnel + pool for the CN
host. The existing public-hostname ingress on the US tunnel is untouched.

## Why this shape (not a plain two-origin Load Balancer)

Both hosts sit behind NAT with no public IP (gpu4 is reached only via
`cloudflared tunnel run --token ...`, same as the README's production setup).
A Cloudflare Load Balancer origin has to be an address Cloudflare's edge can
reach directly -- it can't dial into a NATed host's own outbound tunnel. The
supported way to bridge that is Cloudflare's **Private Network Load
Balancing**: each host's tunnel gets a Virtual Network + a private IP route,
and the LB pool origin references that private IP + virtual network instead of
a public IP.

## What Terraform manages vs. what you still do by hand

Terraform (this module):
- CN tunnel object, both Virtual Networks, both private IP routes
- Health monitor, two pools (`embed-us`, `embed-cn`)
- The `embed.preseen.ai` load balancer with
  `steering_policy = "least_outstanding_requests"` over
  `default_pools = [pool-us, pool-cn]`, `fallback_pool = pool-us`

Still manual:
1. Provision the CN server and get the embedding-provider instance running on
   it (same host-venv flow as gpu4 -- see main `README.md`).
2. Run `terraform apply` to create the CN tunnel object, then fetch its token:
   ```bash
   cloudflared tunnel token <cn tunnel id from `terraform output` / dashboard>
   ```
3. On the CN host, run cloudflared with that token so the private route is
   actually live (this is a *second* cloudflared process from the one that
   would front a public hostname -- private network routing is enabled by the
   route existing server-side, no extra cloudflared flag needed for a
   remotely-managed tunnel).
4. Find the US tunnel's existing UUID (`cloudflared tunnel list` on gpu4, or
   Zero Trust dashboard -> Networks -> Tunnels) and put it in
   `us_tunnel_id` in `terraform.tfvars`. Terraform only attaches a route to it,
   never recreates it.

## Prerequisites / cost

- Cloudflare **Load Balancing** is billed as a separate account-level
  subscription, independent of the zone plan (available self-serve from Free
  through Business; only PoP-level steering is Enterprise-only). Checked via
  the API on 2026-09-15: `preseen.ai` is currently on the **Free Website**
  zone plan -- that does not block buying the Load Balancing add-on, but it
  does mean it isn't already included in anything. Enable it under
  Dashboard -> Manage Account -> Billing -> Load Balancing (or Traffic ->
  Load Balancing on the zone) before `terraform apply`, otherwise the API
  calls fail.

  Monthly cost for exactly this config (2 pools, 1 origin each,
  `least_outstanding_requests` steering, 60s health interval, 1 check region
  per pool -- as configured in `load_balancer.tf`):

  | Item | Cost |
  |---|---|
  | Base subscription (up to 2 origins, 60s interval, 1 check region) | $5/mo |
  | `least_outstanding_requests` steering | included in base -- Cloudflare's published a-la-carte pricing only calls out `geo` steering (+$10/mo) and session affinity as extra; least-connections/least-outstanding-requests/random are not listed as chargeable extras. **Not independently confirmed against a live invoice** -- verify in Dashboard -> Billing after enabling, since this specific line isn't spelled out in the public docs the way the geo surcharge is. |
  | DNS queries to `embed.preseen.ai` | first 500K/mo free (shared across the account's load balancers), then $0.50 per additional 500K |
  | **Expected total, before query overage** | **~$5/mo** (pending the billing confirmation above) |

  Cost goes up if you later change the module's defaults:
  - Switching `steering_policy` to `"geo"`: +$10/mo.
  - `interval = 30` on the monitor: +$10/mo. `interval = 15`: +$15/mo.
  - More than 1 `check_regions` entry per pool: +$10/mo for up to 4 total, +$15/mo for up to 8.
  - More than 2 origins total across pools: +$5/mo per additional origin.

  These figures come from Cloudflare's published Load Balancing add-on pricing
  structure; confirm current numbers in Dashboard -> Billing before relying on
  them, since Cloudflare can change pricing without notice.

- Private Network routing needs **Cloudflare Zero Trust (Cloudflare One)**
  enabled on the account. The free Zero Trust tier is enough for this -- it's
  not gated on seat count for this use case.
- No ICP license or China-based entity needed for this specific setup:
  `least_outstanding_requests` steering only compares in-flight request counts
  per pool, it doesn't change how mainland clients reach the Cloudflare edge
  itself.
- API token permissions needed: `Zone:DNS:Edit`, `Account:Load Balancing:
  Monitors and Pools:Edit`, `Account:Load Balancers:Edit`, `Account:Cloudflare
  Tunnel:Edit`, `Account:Cloudflare One Networks:Edit`. The token currently in
  `preseen-ai-gateway/cf-api.env` only has `dns_records:edit/read` +
  `zone:read` -- it cannot create the Load Balancer objects in this module, a
  broader-scoped token is needed for `terraform apply`.

## Usage

```bash
cp terraform.tfvars.example terraform.tfvars
# fill in account_id, zone_id, us_tunnel_id, cn_tunnel_secret (openssl rand -base64 32)
export CLOUDFLARE_API_TOKEN=...
terraform init
terraform plan
terraform apply
```

## Verification

```bash
curl -s https://embed.preseen.ai/healthz
```

Since routing is now load-based (`least_outstanding_requests`), not
geographic, a single client can't predict which pool a given request lands on
by its own location -- both pools are eligible for every request, and the
choice depends on which pool has fewer in-flight requests at that moment. To
actually see steering behave differently between the pools, drive enough
concurrent load at `/v1/embeddings` to make one pool's queue depth diverge
from the other's (e.g. saturate one instance's `MAX_BATCH_SIZE`/CPU-GPU
adaptive batching from the main `README.md`) and watch requests shift toward
whichever pool is still keeping up.

Check `pool_us`/`pool_cn` health and request distribution in the Cloudflare
dashboard under Load Balancing -> Analytics, or via the GraphQL Analytics API.

## Rollback

`steering_policy = "off"` with only `default_pools = [pool-us]` collapses this
back to a single-origin setup without touching DNS, if the CN origin needs to
be pulled out of rotation.
