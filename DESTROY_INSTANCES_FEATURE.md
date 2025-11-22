# Instance Destruction Feature

## Overview

This feature adds comprehensive instance management capabilities to the Vast.ai CLI, with automatic emergency cleanup on insufficient credit errors.

## New CLI Commands

### 1. **`imggenhub-vast instances`** — List all active instances

View all currently running instances with costs and SSH details:

```bash
poetry run imggenhub-vast instances
```

**Output:**
```
Instance ID     GPU                  Status          Cost/hr SSH
  28116358     Tesla P40             running          $0.1100 root@183.89.209.74:56836
  28116359     RTX 3090              running          $0.2500 root@192.168.1.50:42891
```

### 2. **`imggenhub-vast destroy`** — Destroy a specific instance

Stop billing for one instance:

```bash
poetry run imggenhub-vast destroy --instance-id 28116358
```

**Output:**
```
Successfully destroyed instance 28116358
```

### 3. **`imggenhub-vast destroy-all`** — Emergency cleanup

Destroy ALL active instances at once to stop all billing:

```bash
poetry run imggenhub-vast destroy-all
```

**Output:**
```
Found 3 active instance(s). Destroying all...
✓ Destroyed instance 28116358 (Tesla P40)
✓ Destroyed instance 28116359 (RTX 3090)
✓ Destroyed instance 28116360 (RTX 4090)

Destroyed 3/3 instance(s)
```

## Automatic Cleanup on Insufficient Credit

When you encounter an `insufficient_credit` error during `reserve` or `auto` commands:

1. **Error is detected** automatically
2. **All instances are destroyed** immediately to prevent further charges
3. **Clear error message** is shown
4. **Billing is stopped**

### Example:

```bash
$ poetry run imggenhub-vast reserve --offer-id 12345
[ERROR] Insufficient credit error detected. Destroying all instances...
[INFO] Destroying 2 instance(s)...
[INFO] ✓ Destroyed instance 28116358 (Tesla P40)
[INFO] ✓ Destroyed instance 28116359 (RTX 3090)
[ERROR] All instances destroyed to prevent further costs.
[ERROR] Command failed: insufficient_credit: Your account lacks credit
```

## Implementation Details

### API Changes

- **`destroy_instance(instance_id)`** — Now handles empty JSON responses (API returns 204 or empty body on success)
- **`list_instances()`** — Already filters only rentable offers

### CLI Changes

- Added `handle_destroy()` handler for single instance destruction
- Added `handle_destroy_all()` handler for emergency cleanup
- Added `handle_instances()` handler to list active instances
- Added helper functions:
  - `_destroy_all_instances(client)` — Destroys all instances from a client
  - `_list_instances(client)` — Formats and prints active instances

### Orchestrator Changes

- Enhanced `rent_cheapest()` to catch `insufficient_credit` errors and trigger emergency cleanup
- Added `_destroy_all_instances()` method for emergency cleanup

### Error Handling

- Improved error message formatting to include both error type and message from API
- Graceful handling of API failures during cleanup
- Individual instance destruction failures don't prevent cleanup of remaining instances

## Workflow Examples

### Example 1: Check costs before destroying

```bash
# See what's running
poetry run imggenhub-vast instances

# Destroy specific instance
poetry run imggenhub-vast destroy --instance-id 28116358
```

### Example 2: Emergency cleanup on unexpected charges

```bash
# Destroy everything immediately
poetry run imggenhub-vast destroy-all
```

### Example 3: Automatic cleanup (triggered automatically)

```bash
# Auto-deploy but account has insufficient credit
poetry run imggenhub-vast auto --prompt "test" --max-hourly-price 0.20

# ERROR: Insufficient credit error detected. Destroying all instances...
# ✓ Destroyed instance 28116358 (Tesla P40)
# All instances destroyed to prevent further costs.
```

## Cost Prevention Strategy

1. **Always check costs**: Use `imggenhub-vast instances` to see active instances
2. **Destroy immediately after use**: Don't leave instances running idle
3. **Monitor billing**: Vast.ai charges by the hour, even during idle time
4. **Use `--no-spot` for predictable costs**: Spot instances can be cheaper but may be interrupted
5. **Emergency nuclear option**: Use `destroy-all` if you see unexpected charges

## Testing

All new functionality has been tested and validated:

- ✅ `instances` command displays active instances correctly
- ✅ `destroy` command destroys single instance successfully
- ✅ `destroy-all` command destroys all instances at once
- ✅ Automatic cleanup triggers on `insufficient_credit` error
- ✅ API handles empty JSON responses gracefully
- ✅ Error messages include full context (type + message)

## Files Modified

1. `src/imggenhub/vast_ai/cli/__init__.py` — Added 3 new command handlers
2. `src/imggenhub/vast_ai/api/__init__.py` — Improved `destroy_instance()` error handling
3. `src/imggenhub/vast_ai/auto_deploy.py` — Added emergency cleanup on credit errors
4. `README.md` — Documented destroy commands and automatic cleanup
5. `VAST_AI_CLI_GUIDE.md` — Added complete guide for destroy commands
