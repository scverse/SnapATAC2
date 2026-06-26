Writing AI-Optimized Documentation
==================================

## Intent Boundaries & Behavioral Constraints

Do not rely on the AI to infer the structural limits of your package from raw logic. It must be explicitly told what paths to take and avoid.

- Action-Oriented Explanations: Write documentation using a direct, imperative style: "To achieve X, execute Y." Avoid abstract architectural philosophy; prioritize clear, sequential execution steps.
- Explicit Anti-Patterns: Include a dedicated, highly visible section named Common Mistakes or Anti-Patterns. Tell the AI exactly what not to do, especially where your package deviates from standard industry conventions.
Example: > ❌ Anti-Pattern: Do not instantiate the User class directly (e.g., user = User()).
Correct Pattern: Always provision users through the client factory using client.users.create().

## Structural Sufficiency in Code Examples

Isolated snippet fragments cause AIs to omit dependencies or guess configuration setups. Every code example must be an independent, functional ecosystem.

- End-to-End Runnability: Every code example must be a complete, copy-pasteable script that executes successfully without modifications.
- Mandatory Initializations: Include all necessary top-level import statements and explicit client initialization routines (e.g., loading environment variables or passing API keys).

## Examples

```python
import os
from typing import Literal, Dict, Any
from my_package.exceptions import ResourceNotFoundError, AuthenticationError

def manage_container_lifecycle(
    container_id: str, 
    action: Literal["start", "stop", "restart"]
) -> Dict[str, Any]:
    """
    Execute lifecycle commands on a specific cloud infrastructure container.
    
    Use this tool when an agent or script needs to explicitly alter the operational 
    state of a running or stopped compute container. 

    Anti-Patterns
    -------------
    - Do NOT use this function to check container health or metrics; use 
      `get_container_metrics` instead to save compute tokens.
    - Do NOT pass raw arbitrary strings to `action`. Only the explicit literal
      values defined in the type signature are accepted.

    Parameters
    ----------
    container_id : str
        The unique alphanumeric identifier of the target container 
        (e.g., "ctnr-8f9d2").
    action : {"start", "stop", "restart"}
        The structural mutation to apply. Must match one of the allowed literals.

    Returns
    -------
    dict of (str, Any)
        A structural summary of the post-action state.
        Contains 'container_id' (str), 'status' (str), and 'code' (int).

    Raises
    ------
    ResourceNotFoundError
        If the provided container_id does not exist in the infrastructure registry.
    AuthenticationError
        If the client token is missing, unauthorized, or expired.

    See Also
    --------
    get_container_metrics : Fetch telemetry without altering state.

    Examples
    --------
    >>> import os
    >>> from my_package.client import InfrastructureClient
    >>> from my_package.exceptions import ResourceNotFoundError, AuthenticationError
    
    1. Setup and authenticate the primary client
    >>> token = os.environ.get("INFRA_API_TOKEN")
    >>> client = InfrastructureClient(api_token=token)
    
    2. Execute workflow with explicit, resilient error handling
    >>> try:
    ...     response = client.manage_container_lifecycle(
    ...         container_id="ctnr-8f9d2", 
    ...         action="restart"
    ...     )
    ...     print(f"Success! New state: {response['status']}")
    ... except ResourceNotFoundError:
    ...     print("Target container missing. Initiating system sync fallback...")
    ...     client.sync_inventory()
    ... except AuthenticationError:
    ...     print("Critical failure: Check environment variable token.")
    """
    pass
```