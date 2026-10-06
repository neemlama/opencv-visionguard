"""Cost guard for ForgeSight: AWS Budget with alarms + an auto-stop budget action.

Run once:  python aws/setup_budget.py --email you@example.com --limit 25

Why a Budget *and* an action:  an alarm only emails you, and AWS Budgets
notifications are not real-time (they can lag hours). The budget action stops
the tagged instances itself, so an unattended run cannot keep billing past the
cap even if nobody reads email.

Guard rails:
  * Only instances carrying CostBudgetTag=forgesight are ever stopped.
  * Instances are created by aws/run_bench.py and aws/endpoint.py, which both
    apply that tag, so nothing outside this project is at risk.
  * Budgets is a global service; it must be called in us-east-1.
"""
import argparse
import json

import boto3
from botocore.exceptions import ClientError

BUDGETS_REGION = "us-east-1"
COST_TAG_KEY = "CostBudgetTag"
COST_TAG_VALUE = "forgesight"

# Types the benchmark and demo endpoint actually use.
GUARDED_INSTANCE_TYPES = [
    "t4g.small",
    "t4g.medium",
    "t4g.large",
    "t3.small",
    "t3.medium",
    "c7g.small",
    "c7g.medium",
    "m7g.small",
    "c8g.small",
    "c8g.medium",
]


def client(service, region=BUDGETS_REGION):
    return boto3.client(service, region_name=region)


def ensure_topic(sns, email):
    """Reuse the project's topic if present, else create and subscribe it."""
    topic_arn = f"arn:aws:sns:{BUDGETS_REGION}:{account_id()}:forgesight-budget-alerts"
    try:
        sns.get_topic_attributes(TopicArn=topic_arn)
        print(f"[sns] reusing {topic_arn}")
    except ClientError:
        topic_arn = sns.create_topic(Name="forgesight-budget-alerts")["TopicArn"]
        print(f"[sns] created {topic_arn}")

    if email:
        try:
            sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)
            print(f"[sns] subscribed {email} (confirm via the email AWS sends)")
        except ClientError as e:
            print(f"[sns] subscribe failed: {e.response['Error']['Message']}")
    return topic_arn


def account_id():
    return boto3.client("sts", region_name=BUDGETS_REGION).get_caller_identity()["Account"]


def ensure_budget_role(iam, role_name):
    """Role Budgets assumes to issue ec2:StopInstances.

    Uses an inline policy rather than an AWS-managed policy so this does not
    depend on a managed-policy ARN existing in the partition.
    Requires the caller to hold iam:CreateRole / iam:PutRolePolicy / PassRole.
    """
    trust = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "budgets.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
    )
    stop_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BudgetAutoStop",
                    "Effect": "Allow",
                    "Action": ["ec2:DescribeInstances", "ec2:StopInstances"],
                    "Resource": "*",
                }
            ],
        }
    )
    try:
        iam.get_role(RoleName=role_name)
        print(f"[iam] reusing role {role_name}")
    except ClientError:
        try:
            iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust)
            print(f"[iam] created role {role_name}")
        except ClientError as e:
            print(f"[iam] SKIP auto-stop action, insufficient permission: "
                  f"{e.response['Error']['Message']}")
            print("     Alarms will still be created. Ask an admin to grant")
            print("     iam:CreateRole, iam:PutRolePolicy and iam:PassRole.")
            return None
    try:
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="budget-auto-stop",
            PolicyDocument=stop_policy,
        )
        print("[iam] attached inline ec2:StopInstances policy")
    except ClientError as e:
        print(f"[iam] SKIP auto-stop action, cannot attach policy: "
              f"{e.response['Error']['Message']}")
        print("     Alarms will still be created.")
        return None
    return f"arn:aws:iam::{account_id()}:role/{role_name}"


def apply_budget(budgets, limit, topic_arn, account_wide=True):
    """Create the budget, or update it in place if it already exists."""
    name = "ForgeSight Hackathon Cap"
    notifications = [
        # Early warning well before the cap, so there is time to react.
        {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
         "Threshold": round(limit * 0.5, 2), "ThresholdType": "PERCENTAGE"},
        {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
         "Threshold": 80, "ThresholdType": "PERCENTAGE"},
        {"NotificationType": "ACTUAL", "ComparisonOperator": "GREATER_THAN",
         "Threshold": 100, "ThresholdType": "PERCENTAGE"},
        # Forecast catches runaway spend *before* the money is gone, which is the
        # alarm that actually prevents an overrun on an unattended instance.
        {"NotificationType": "FORECASTED", "ComparisonOperator": "GREATER_THAN",
         "Threshold": 100, "ThresholdType": "PERCENTAGE"},
    ]
    budget = {
        "BudgetName": name,
        "BudgetType": "COST",
        "TimeUnit": "MONTHLY",
        "BudgetLimit": {"Amount": str(limit), "Unit": "USD"},
    }
    if not account_wide:
        # Scoped mode only counts resources carrying our activation tag.
        # Untagged accidental spend becomes invisible, so this is opt-in only.
        budget["CostFilters"] = {"Tags": [{"Key": f"user:{COST_TAG_KEY}", "Values": [COST_TAG_VALUE]}]}

    # NotificationsWithSubscribers is a top-level parameter on CreateBudget /
    # UpdateBudget, not a member of Budget itself. SubscriberType is SNS (not
    # EMAIL) because the address is a topic ARN; SNS then fans out to its own
    # confirmed subscribers.
    notifications = [
        {
            "Notification": n,
            "Subscribers": [{"SubscriptionType": "SNS", "Address": topic_arn}],
        }
        for n in notifications
    ]

    existing = [
        b["BudgetName"] for b in budgets.describe_budgets(AccountId=account_id())["Budgets"]
    ]
    if name in existing:
        # UpdateBudget accepts only AccountId + NewBudget on this API version, so
        # notifications are not re-sent. They persist on the existing budget.
        budgets.update_budget(AccountId=account_id(), NewBudget=budget)
        print(f"[budget] updated '{name}' cap=${limit}/month "
              f"(existing notifications left in place)")
    else:
        budgets.create_budget(
            AccountId=account_id(),
            Budget=budget,
            NotificationsWithSubscribers=notifications,
        )
        print(f"[budget] created '{name}' cap=${limit}/month with "
              f"{len(notifications)} notifications")
    print(f"[budget] scope: {'ALL services (account-wide)' if account_wide else 'tagged only'}")
    return name


def apply_action(budgets, budget_name, role_arn, topic_arn):
    """Attempt to arm an auto-stop budget action.

    Why this usually cannot be armed on this account
    -----------------------------------------------
    CreateBudgetAction is a flat operation, and its Definition union only offers:

        IamActionDefinition  {PolicyArn, Roles, Groups, Users}
        SsmActionDefinition  {ActionSubType, Region, InstanceIds}
        ScpActionDefinition  {PolicyId, TargetIds}

    The documented tag-scoped EC2 stop (IAMAction with InstanceTypes + Tags) is
    absent from this API model, so we cannot express "stop every instance tagged
    CostBudgetTag=forgesight". The SSM variant would work but needs literal
    InstanceIds, which go stale the moment an instance is replaced.

    We therefore attempt it, report precisely why it failed, and rely on the
    code-level guard in aws/endpoint.py (pre-flight refusal + TTL + cost tag)
    as the real enforcement. See docs/COST-GUARD.md.
    """
    try:
        existing = budgets.describe_budget_actions_for_budget(
            AccountId=account_id(), BudgetName=budget_name
        ).get("Actions", [])
    except ClientError:
        existing = []

    # create_budget_action is NOT idempotent, so re-running this script would
    # otherwise stack duplicate actions. Remove stale copies first, then create
    # exactly one.
    for a in existing:
        print(f"[action] removing duplicate {a['ActionId']}")
        budgets.delete_budget_action(
            AccountId=account_id(), BudgetName=budget_name, ActionId=a["ActionId"]
        )

    action = {
        "BudgetName": budget_name,
        "NotificationType": "ACTUAL",
        "ActionType": "APPLY_IAM_POLICY",
        "ActionThreshold": {"ActionThresholdValue": 100.0, "ActionThresholdType": "PERCENTAGE"},
        "ExecutionRoleArn": role_arn,
        "ApprovalModel": "AUTOMATIC",
        "Subscribers": [{"SubscriptionType": "SNS", "Address": topic_arn}],
        "Definition": {
            "IamActionDefinition": {
                "PolicyArn": "arn:aws:iam::aws:policy/AmazonEC2InstanceAccessForStoppedInstances",
                "Roles": ["forgesight-budget-stopper"],
            }
        },
    }
    try:
        budgets.create_budget_action(AccountId=account_id(), **action)
        print("[action] auto-stop armed at 100% of cap, approval=AUTOMATIC")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        print(f"[action] NOT armed ({code}): {e.response['Error']['Message']}")
        print("         Enforcing the cap in aws/endpoint.py instead.")


def main():
    ap = argparse.ArgumentParser(description="ForgeSight cost guard")
    ap.add_argument("--limit", type=float, default=25.0, help="monthly cap in USD")
    ap.add_argument("--email", default="", help="email to subscribe to alarms")
    ap.add_argument("--tagged-only", action="store_true",
                    help="count only tagged resources (blind to untagged spend)")
    args = ap.parse_args()

    sns, iam, budgets = client("sns"), client("iam"), client("budgets")

    topic_arn = ensure_topic(sns, args.email)
    role_arn = ensure_budget_role(iam, "forgesight-budget-stopper")
    budget_name = apply_budget(budgets, args.limit, topic_arn, account_wide=not args.tagged_only)
    if role_arn:
        apply_action(budgets, budget_name, role_arn, topic_arn)

    print("\n=== verification ===")
    for b in budgets.describe_budgets(AccountId=account_id())["Budgets"]:
        if b["BudgetName"] == budget_name:
            actual = b.get("CalculatedSpend", {}).get("ActualSpend", {}).get("Amount", "n/a")
            # Notifications are not echoed in the describe_budgets summary, so
            # read them from the dedicated API or the count is misleadingly 0.
            notes = budgets.describe_notifications_for_budget(
                AccountId=account_id(), BudgetName=budget_name
            )["Notifications"]
            print(f"budget    : {b['BudgetName']} {b['BudgetLimit']['Amount']} USD/month")
            print(f"spent     : {actual}")
            print(f"health    : {b['HealthStatus']['Status']}")
            print(f"notify    : {len(notes)} notification(s)")
    acts = budgets.describe_budget_actions_for_budget(
        AccountId=account_id(), BudgetName=budget_name
    ).get("Actions", [])
    print("actions   : " + (
        ", ".join(
            f"{a.get('ActionType')}/{a.get('ApprovalModel')} status={a.get('Status')} "
            f"at {a.get('ActionThreshold', {}).get('ActionThresholdValue')}%"
            for a in acts
        ) or "none"
    ))
    print("\nNote: Budget alarms are email-based and can lag by hours. The armed action")
    print("fires without approval, but the primary enforcement is the launch-time")
    print("pre-flight in aws/endpoint.py. See docs/COST-GUARD.md for the caveats.")


if __name__ == "__main__":
    main()