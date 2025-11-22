# Instance Destruction - Complete Usage Guide

## Quick Reference

```bash
# View all active instances
poetry run imggenhub-vast instances

# Destroy one instance
poetry run imggenhub-vast destroy --instance-id 28116358

# Destroy ALL instances (emergency cleanup)
poetry run imggenhub-vast destroy-all
```

## Common Scenarios

### Scenario 1: You rented an instance but don't need it anymore

```bash
# 1. See what you have running
$ poetry run imggenhub-vast instances
Instance ID     GPU                  Status          Cost/hr SSH
  28116358     Tesla P40             running          $0.1100 root@183.89.209.74:56836

# 2. Destroy it to stop billing
$ poetry run imggenhub-vast destroy --instance-id 28116358
Successfully destroyed instance 28116358

# 3. Verify it's gone
$ poetry run imggenhub-vast instances
No active instances.
```

### Scenario 2: Multiple instances running and you want to destroy all

```bash
# See all running instances
$ poetry run imggenhub-vast instances
Instance ID     GPU                  Status          Cost/hr SSH
  28116358     Tesla P40             running          $0.1100 root@183.89.209.74:56836
  28116359     RTX 3090              running          $0.2500 root@192.168.1.50:42891
  28116360     RTX 4090              running          $0.5000 root@10.0.0.50:33891

# Destroy them all at once
$ poetry run imggenhub-vast destroy-all
Found 3 active instance(s). Destroying all...
✓ Destroyed instance 28116358 (Tesla P40)
✓ Destroyed instance 28116359 (RTX 3090)
✓ Destroyed instance 28116360 (RTX 4090)

Destroyed 3/3 instance(s)
```

### Scenario 3: Account runs out of credit (automatic cleanup)

```bash
# You try to rent a GPU but your account has insufficient credit
$ poetry run imggenhub-vast reserve --offer-id 12345
[ERROR] Insufficient credit error detected. Destroying all instances...
[INFO] Destroying 2 instance(s)...
[INFO] ✓ Destroyed instance 28116358 (Tesla P40)
[INFO] ✓ Destroyed instance 28116359 (RTX 3090)
[INFO] All instances destroyed to prevent further costs.
[ERROR] Command failed: insufficient_credit: Your account lacks credit; see the billing page.
```

**What happened:**
- The system detected you ran out of credit
- It automatically destroyed all your instances to prevent further charges
- You need to add more credit and try again

### Scenario 4: Using auto-deploy with automatic cleanup on failure

```bash
# Start an auto-deployment that will fail if credit runs out
$ poetry run imggenhub-vast auto \
    --prompt "test image" \
    --max-hourly-price 0.50 \
    --steps 30

# If insufficient credit is detected:
# 1. All instances are automatically destroyed
# 2. An error message is displayed
# 3. You can add credit and retry
```

## Cost Control Strategies

### Strategy 1: Always check before destroying

```bash
# See the cost per hour of each instance
poetry run imggenhub-vast instances

# Calculate total cost
# Example: 3 instances at $0.11, $0.25, $0.50 = $0.86/hour
# After 1 hour, that's $0.86; after 24 hours, that's $20.64
```

### Strategy 2: Destroy immediately after use

```bash
# Don't leave instances running idle - they continue to bill!
poetry run imggenhub-vast destroy --instance-id 28116358
```

### Strategy 3: Use spot instances for savings

```bash
# Spot instances can be 30-70% cheaper
poetry run imggenhub-vast list --limit 10
# Spot instances will be marked as cheaper

# Reserve spot instances (enabled by default unless --no-spot)
poetry run imggenhub-vast auto --prompt "test" --max-hourly-price 0.20
```

### Strategy 4: Monitor your account regularly

```bash
# Check what's running
poetry run imggenhub-vast instances

# Go to Vast.ai dashboard to see current credits and spending
# https://console.vast.ai/billing/
```

## Emergency Response Plan

### If you see unexpected charges:

```bash
# STEP 1: Stop all billing immediately
poetry run imggenhub-vast destroy-all

# STEP 2: Check what was running
poetry run imggenhub-vast instances  # Should show "No active instances"

# STEP 3: Add more credit to your account
# https://console.vast.ai/billing/

# STEP 4: Try your operation again carefully
poetry run imggenhub-vast list --limit 5  # Check offers first
poetry run imggenhub-vast reserve --offer-id 12345  # Rent one instance
```

## API Details

### destroy_instance(instance_id: int) -> bool

Destroys a single instance and stops all billing for that instance.

**Parameters:**
- `instance_id` (int): The instance ID to destroy

**Returns:**
- `True` if successfully destroyed
- `False` if destruction failed

**Errors:**
- Raises `ValueError` if instance_id is invalid (≤ 0)
- Raises `requests.RequestException` if API call fails

### list_instances() -> List[VastInstance]

Lists all active instances currently running.

**Returns:**
- List of VastInstance objects with details:
  - `id`: Instance ID
  - `gpu_name`: GPU model (e.g., "Tesla P40")
  - `status`: Current status
  - `price_per_hour`: Hourly cost in USD
  - `ssh_host`: SSH hostname
  - `ssh_port`: SSH port
  - `ssh_user`: SSH username (usually "root")

## Billing Examples

### Example 1: P40 for 1 hour
- Price: $0.11/hour
- 1 instance × 1 hour = $0.11

### Example 2: RTX 3090 for 4 hours
- Price: $0.25/hour
- 1 instance × 4 hours = $1.00

### Example 3: RTX 4090 overnight (12 hours)
- Price: $0.50/hour
- 1 instance × 12 hours = $6.00

### Example 4: Multiple instances (worst case!)
- RTX 4090: $0.50/hour
- RTX 3090: $0.25/hour
- Tesla P40: $0.11/hour
- Total: $0.86/hour
- **If left running 24 hours: $20.64 in charges!**

## Prevention Tips

1. ✅ Always use `destroy` when done
2. ✅ Check `instances` before going idle
3. ✅ Set `--max-hourly-price` to limit costs
4. ✅ Use `--keep-instance` only when needed
5. ✅ Monitor your Vast.ai account regularly
6. ✅ Test with `--no-spot` if you prefer predictable costs
7. ❌ Don't leave instances running "just in case"
8. ❌ Don't assume instances will auto-destroy
9. ❌ Don't ignore low-credit warnings

## Troubleshooting

### "Instance not found" when destroying
- Instance may have already been destroyed
- Try `instances` to see what's currently active
- Vast.ai automatically removes destroyed instances

### destroy-all shows 0 instances destroyed
- No instances are currently running (good!)
- Use `instances` to verify

### Destroy command fails with permission error
- Check your API key is valid
- Verify the instance is yours (check Vast.ai dashboard)
- Try the destroy-all command to see if it's an API issue

### Still seeing charges after destroy
- May take a few minutes to propagate
- Check Vast.ai dashboard for confirmation
- Contact Vast.ai support if charges continue
