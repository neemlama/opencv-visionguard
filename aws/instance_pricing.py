"""Authoritative on-demand prices and specs for the benchmark instance types.

Hardcoded rates were the weakest part of the cost argument, and one comparison
was invalid: c8g.medium is 1 vCPU while t3.medium is 2 vCPU, and t3 is a
burstable family, so a short benchmark can run on burst credits that would not
sustain in production.

This pulls real prices from the AWS Price List API and real specs from EC2, so
the cost table can state vCPU count, burstability and rate together.

    python aws/instance_pricing.py
"""
import argparse
import json

import boto3

TYPES = ["c8g.medium", "c8g.large", "c7i.large", "t3.medium", "m6g.medium",
         "c6g.large", "c6g.medium"]


def _first_on_demand_usd(product_json_str):
    """Pull the USD hourly rate out of a Price List entry.

    The Pricing API returns each PriceList item as a JSON *string* whose price
    lives under terms.OnDemand.<term>.priceDimensions[0].pricePerUnit.USD.
    Reading pricePerUnit off the top level (as a first attempt did) throws
    TypeError, because the top level has no such key.
    """
    item = json.loads(product_json_str) if isinstance(product_json_str, str) \
        else product_json_str
    for term in item.get("terms", {}).get("OnDemand", {}).values():
        # priceDimensions is a DICT keyed by SKU, not a list. Iterating it
        # directly yields strings and the next .get() raises AttributeError.
        for dim in term.get("priceDimensions", {}).values():
            usd = dim.get("pricePerUnit", {}).get("USD")
            if usd is not None:
                return float(usd)
    return None


def prices(region="us-east-1"):
    """On-demand Linux price per hour, us-east-1, no savings plan."""
    pr = boto3.client("pricing", region_name="us-east-1")
    out = {}
    for t in TYPES:
        try:
            f = pr.get_products(
                ServiceCode="AmazonEC2",
                Filters=[
                    {"Type": "TERM_MATCH", "Field": "regionCode", "Value": region},
                    {"Type": "TERM_MATCH", "Field": "instanceType", "Value": t},
                    {"Type": "TERM_MATCH", "Field": "operatingSystem", "Value": "Linux"},
                    {"Type": "TERM_MATCH", "Field": "tenancy", "Value": "Shared"},
                    {"Type": "TERM_MATCH", "Field": "preInstalledSw", "Value": "NA"},
                    {"Type": "TERM_MATCH", "Field": "capacitystatus", "Value": "Used"},
                ],
                MaxResults=1,
            )
            out[t] = _first_on_demand_usd(f["PriceList"][0]) if f["PriceList"] else None
        except Exception as e:
            out[t] = f"error: {type(e).__name__}"
    return out


def specs(region="us-east-1"):
    ec2 = boto3.client("ec2", region_name=region)
    out = {}
    for t in TYPES:
        try:
            d = ec2.describe_instance_types(InstanceTypes=[t])["InstanceTypes"][0]
            out[t] = {
                "vcpu": d["VCpuInfo"]["DefaultVCpus"],
                "memory_gib": d["MemoryInfo"]["SizeInMiB"] // 1024,
                # The capability flag, not BurstableInfo (which is absent even
                # for burstable types in this API version). t3.* sets this True.
                "burstable": bool(d.get("BurstablePerformanceSupported")),
                "processor": d["ProcessorInfo"]["SupportedArchitectures"],
            }
        except Exception as e:
            out[t] = {"error": type(e).__name__}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--out", default="bench/instance_pricing.json")
    args = ap.parse_args()

    p = prices(args.region)
    s = specs(args.region)

    print(f"{'type':<12}{'vCPU':>5}{'memGiB':>8}{'burstable':>11}{'USD/hr':>10}"
          f"{'USD/hr/vCPU':>13}")
    for t in TYPES:
        sp = s.get(t, {})
        rate = p.get(t)
        vcpu = sp.get("vcpu")
        per = f"{rate / vcpu:.5f}" if isinstance(rate, float) and vcpu else "-"
        print(f"{t:<12}{str(vcpu):>5}{str(sp.get('memory_gib')):>8}"
              f"{str(sp.get('burstable')):>11}{str(rate):>10}{per:>13}")

    print("\nMATCHED PAIRS (same vCPU, both non-burstable):")
    for a, b in (("c8g.large", "c7i.large"), ("c8g.medium", "t3.medium")):
        sa, sb = s.get(a, {}), s.get(b, {})
        if sa.get("vcpu") == sb.get("vcpu"):
            print(f"  {a} ({sa.get('vcpu')} vCPU, burstable={sa.get('burstable')}) "
                  f"vs {b} ({sb.get('vcpu')} vCPU, burstable={sb.get('burstable')})"
                  f"  -> {'MATCHED' if not (sa.get('burstable') or sb.get('burstable')) else 'NOT matched (burst)'}")
        else:
            print(f"  {a} ({sa.get('vcpu')} vCPU) vs {b} ({sb.get('vcpu')} vCPU) "
                  f"-> NOT vCPU-matched")

    json.dump({"region": args.region, "prices": p, "specs": s},
              open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
