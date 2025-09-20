import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
class Parameters:
    def __init__(self, beta, delta, v, pm, c, T):
        self.beta = beta  # Present bias factor
        self.delta = delta  # Exponential discount factor
        self.v = v  # Value of subscription
        self.pm = pm  # Subscription price
        self.c = c  # Cancellation cost
        self.T = T  # Time horizon (periods 0 to T)

# Define utility function
def get_utility(x_t, action, params, pi=0):
    """
    Calculates immediate utility based on current state and action.
    x_t: Current state (0 = not subscribed, 1 = subscribed)
    action: Action taken ('do nothing', 'subscribe', 'stay', 'cancel')
    params: Parameters object
    pi: Subsidy for cancellation (reduces cost c)
    """
    if x_t == 0:  # Not subscribed
        return 0
    elif x_t == 1:  # Subscribed
        if action == 'stay':
            return params.v - params.pm
        elif action == 'cancel':
            return params.v - params.pm - (params.c - pi)
    return 0  # Default case

# Compute value functions for exponential discounter (planner)
def compute_Wt(params):
    """
    Computes value functions Wt(x) for the exponential discounter using backward induction.
    Returns array Wt[t][x] where t is time, x is state.
    """
    Wt = np.zeros((params.T + 2, 2))  # Extra period for boundary condition
    for t in range(params.T, -1, -1):
        # State x_t = 0
        val_do_nothing_0 = get_utility(0, 'do nothing', params) + params.delta * Wt[t+1][0]
        val_subscribe_0 = get_utility(0, 'subscribe', params) + params.delta * Wt[t+1][1]
        Wt[t][0] = max(val_do_nothing_0, val_subscribe_0)
        # State x_t = 1
        val_stay_1 = get_utility(1, 'stay', params) + params.delta * Wt[t+1][1]
        val_cancel_1 = get_utility(1, 'cancel', params) + params.delta * Wt[t+1][0]
        Wt[t][1] = max(val_stay_1, val_cancel_1)
    return Wt

# Compute value functions and strategies for naive agent
def compute_Vt_and_sigma(params, Wt_planner, pi=0):
    """
    Computes value functions Vt(x) and strategies sigma_t(x) for the naive agent.
    Wt_planner: Planner's value function (naive agent's belief about future self)
    pi: Subsidy for cancellation
    Returns Vt (value functions) and sigma (strategies)
    """
    Vt = np.zeros((params.T + 2, 2))
    sigma = [['' for _ in range(2)] for _ in range(params.T + 1)]
    for t in range(params.T, -1, -1):
        # State x_t = 0
        perceived_val_do_nothing_0 = get_utility(0, 'do nothing', params) + params.beta * params.delta * Wt_planner[t+1][0]
        perceived_val_subscribe_0 = get_utility(0, 'subscribe', params) + params.beta * params.delta * Wt_planner[t+1][1]
        sigma[t][0] = 'subscribe' if perceived_val_subscribe_0 >= perceived_val_do_nothing_0 else 'do nothing'
        Vt[t][0] = get_utility(0, sigma[t][0], params) + params.beta * params.delta * Vt[t+1][1 if sigma[t][0] == 'subscribe' else 0]
        # State x_t = 1
        perceived_val_stay_1 = get_utility(1, 'stay', params) + params.beta * params.delta * Wt_planner[t+1][1]
        perceived_val_cancel_1 = get_utility(1, 'cancel', params, pi) + params.beta * params.delta * Wt_planner[t+1][0]
        sigma[t][1] = 'stay' if perceived_val_stay_1 >= perceived_val_cancel_1 else 'cancel'
        Vt[t][1] = get_utility(1, sigma[t][1], params, pi) + params.beta * params.delta * Vt[t+1][1 if sigma[t][1] == 'stay' else 0]
    return Vt, sigma

# Calculate welfare gap between planner and naive agent
def calculate_welfare_gap(Wt_planner, Vt_naive, initial_state=1):
    """
    Calculates the welfare gap at t=0 for a given initial state.
    """
    return Wt_planner[0][initial_state] - Vt_naive[0][initial_state]

# Identify procrastination periods
def get_procrastination_periods(params, sigma_naive, sigma_planner):
    """
    Identifies periods where naive agent procrastinates (stays) but planner cancels.
    """
    procrastination_periods = []
    for t in range(params.T + 1):
        if sigma_planner[t][1] == 'cancel' and sigma_naive[t][1] == 'stay':
            procrastination_periods.append(t)
    return procrastination_periods

# Calculate optimal uniform subsidy
def calculate_optimal_subsidy(params, Wt_planner):
    """
    Calculates the optimal subsidy pi* to align naive agent's behavior with planner's.
    """
    max_pi_needed = 0
    for t in range(params.T + 1):
        delta_Wt_future = Wt_planner[t+1][0] - Wt_planner[t+1][1]
        if params.delta * delta_Wt_future > params.c and params.beta * params.delta * delta_Wt_future <= params.c:
            pi_needed = (1 - params.beta) * params.delta * delta_Wt_future
            max_pi_needed = max(max_pi_needed, pi_needed)
    return max_pi_needed

# Main execution block
if __name__ == "__main__":
    # Set baseline parameters
    base_params = Parameters(beta=0.7, delta=0.95, v=5, pm=6, c=0.8, T=10)

    # Compute planner's value function
    Wt_planner = compute_Wt(base_params)

    # Compute planner's optimal strategy
    sigma_planner = [['' for _ in range(2)] for _ in range(base_params.T + 1)]
    for t in range(base_params.T, -1, -1):
        val_do_nothing_0 = get_utility(0, 'do nothing', base_params) + base_params.delta * Wt_planner[t+1][0]
        val_subscribe_0 = get_utility(0, 'subscribe', base_params) + base_params.delta * Wt_planner[t+1][1]
        sigma_planner[t][0] = 'subscribe' if val_subscribe_0 >= val_do_nothing_0 else 'do nothing'
        val_stay_1 = get_utility(1, 'stay', base_params) + base_params.delta * Wt_planner[t+1][1]
        val_cancel_1 = get_utility(1, 'cancel', base_params) + base_params.delta * Wt_planner[t+1][0]
        sigma_planner[t][1] = 'stay' if val_stay_1 >= val_cancel_1 else 'cancel'

    # Baseline simulation (no subsidy)
    Vt_naive_baseline, sigma_naive_baseline = compute_Vt_and_sigma(base_params, Wt_planner, pi=0)
    print("--- Baseline Simulation Results (No Subsidy) ---")
    print(f"Planner's Initial Welfare: {Wt_planner[0][1]:.3f} utility units")
    print(f"Naive Agent's Initial Welfare: {Vt_naive_baseline[0][1]:.3f} utility units")
    welfare_gap_baseline = calculate_welfare_gap(Wt_planner, Vt_naive_baseline)
    print(f"Welfare Gap: {welfare_gap_baseline:.3f} utility units")
    procrast_periods = get_procrastination_periods(base_params, sigma_naive_baseline, sigma_planner)
    print(f"Procrastination Periods: {procrast_periods}")

    # Calculate and apply optimal subsidy
    optimal_pi = calculate_optimal_subsidy(base_params, Wt_planner)
    print(f"\nOptimal Subsidy (pi*): {optimal_pi:.3f} utility units")
    Vt_naive_subsidy, sigma_naive_subsidy = compute_Vt_and_sigma(base_params, Wt_planner, pi=optimal_pi)
    print("\n--- Simulation Results with Optimal Subsidy ---")
    print(f"Naive Agent's Initial Welfare (with Subsidy): {Vt_naive_subsidy[0][1]:.3f} utility units")
    welfare_gap_subsidy = calculate_welfare_gap(Wt_planner, Vt_naive_subsidy)
    print(f"Welfare Gap (with Subsidy): {welfare_gap_subsidy:.3f} utility units")
    procrast_periods_subsidy = get_procrastination_periods(base_params, sigma_naive_subsidy, sigma_planner)
    print(f"Procrastination Periods (with Subsidy): {procrast_periods_subsidy}")

    # Sensitivity Analysis: Varying Beta
    print("\n--- Sensitivity Analysis: Varying Beta ---")
    betas = np.linspace(0.1, 1.0, 20)
    welfare_gaps_beta = []
    for b in betas:
        temp_params = Parameters(beta=b, delta=base_params.delta, v=base_params.v,
                                 pm=base_params.pm, c=base_params.c, T=base_params.T)
        temp_Wt_planner = compute_Wt(temp_params)
        temp_Vt_naive, _ = compute_Vt_and_sigma(temp_params, temp_Wt_planner, pi=0)
        welfare_gaps_beta.append(calculate_welfare_gap(temp_Wt_planner, temp_Vt_naive))
    plt.figure(figsize=(10, 6))
    plt.plot(betas, welfare_gaps_beta, marker='o', linestyle='-')
    plt.title('Welfare Gap vs. Present Bias (Beta)')
    plt.xlabel('Beta')
    plt.ylabel('Welfare Gap (Utility Units)')
    plt.grid(True)
    plt.savefig('welfare_gap_beta.png')
    plt.close()

    # Sensitivity Analysis: Varying Cancellation Cost
    print("\n--- Sensitivity Analysis: Varying Cancellation Cost ---")
    c_values = np.linspace(0.1, 2.0, 20)
    welfare_gaps_c = []
    for c_val in c_values:
        temp_params = Parameters(beta=base_params.beta, delta=base_params.delta, v=base_params.v,
                                 pm=base_params.pm, c=c_val, T=base_params.T)
        temp_Wt_planner = compute_Wt(temp_params)
        temp_Vt_naive, _ = compute_Vt_and_sigma(temp_params, temp_Wt_planner, pi=0)
        welfare_gaps_c.append(calculate_welfare_gap(temp_Wt_planner, temp_Vt_naive))
    plt.figure(figsize=(10, 6))
    plt.plot(c_values, welfare_gaps_c, marker='o', linestyle='-')
    plt.title('Welfare Gap vs. Cancellation Cost')
    plt.xlabel('Cancellation Cost (c)')
    plt.ylabel('Welfare Gap (Utility Units)')
    plt.grid(True)
    plt.savefig('welfare_gap_c.png')
    plt.close()

    # Verification test for beta = 1
    print("\n--- Verification Test: Beta = 1 ---")
    test_params = Parameters(beta=1.0, delta=base_params.delta, v=base_params.v,
                             pm=base_params.pm, c=base_params.c, T=base_params.T)
    test_Wt = compute_Wt(test_params)
    test_Vt, _ = compute_Vt_and_sigma(test_params, test_Wt, pi=0)
    test_gap = calculate_welfare_gap(test_Wt, test_Vt)
    print(f"Welfare Gap for Beta = 1: {test_gap:.3f} utility units (should be ~0)")

    print("\nSimulation complete. Check 'welfare_gap_beta.png' and 'welfare_gap_c.png' for plots.")
